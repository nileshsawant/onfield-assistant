/**
 * SLURM lifecycle + `ofa --serve` bring-up.
 *
 * The extension host runs on the Kestrel login node (case a), so we
 * spawn the equivalent of `bin/ofa`'s startup pipeline as a child
 * process:
 *
 *   salloc … srun --pty ofa --serve --serve-quiet [--serve-enable-tools]
 *
 * Stream its stderr, parse the printed connection block, and resolve
 * once we've extracted (jobId, node, port, token) AND seen the
 * `[ofa-serve] Ctrl+C to stop.` readiness sentinel.
 *
 * Killing the child cascades SIGTERM through salloc → srun → ofa,
 * which releases the SLURM allocation cleanly. `scancel <jobId>` is a
 * belt-and-braces fallback for the case where the child died on us but
 * the job is still around.
 */
import * as cp from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { promisify } from 'node:util';
import type { Logger } from './logger';

const execFile = promisify(cp.execFile);

export interface OfaEndpoint {
    /** Compute-node hostname (e.g. 'x3100c0s5b0n0', 'kl3'). */
    node: string;
    /** TCP port ofa is listening on inside the compute node. */
    port: number;
    /** Bearer token from the printed 'apiKey = …' banner line. */
    token: string;
    /** Convenience: `http://<node>:<port>/v1`. */
    baseUrl: string;
    /** SLURM job id (from `salloc: Granted job allocation …`). */
    jobId: string;
    /** Long-running child process; kill to release SLURM. `null` for
     *  endpoints we adopted from a pre-existing SLURM job (see
     *  adoptExisting.ts) — in that case disconnect() falls back to
     *  `scancel <jobId>` alone. */
    process: cp.ChildProcess | null;
}

export interface SlurmOptions {
    /** Empty string = let bin/ofa's sacctmgr auto-detection run. */
    account: string;
    partition: string;
    walltime: string;
    gres: string;
    enableTools: boolean;
    /** Empty string = don't set OFA_MODEL, let bin/ofa use its own
     *  default (currently gemma4:31b). Any non-empty value is passed
     *  through as-is; validation against MODEL_REGISTRY is done by
     *  bin/ofa itself so this stays forward-compatible with entries
     *  added via $OFA_ROOT/models.json / $OFA_MODELS_JSON. */
    model: string;
    /** Absolute path to bin/ofa. Resolved by resolveOfaBin() before
     *  connect() is called. Passing this explicitly (rather than
     *  relying on $PATH inside the extension host) is required
     *  because VS Code's Remote-SSH server does not source ~/.bashrc,
     *  so any `module load assistant` a user runs manually in a
     *  terminal never propagates to the extension host process. */
    ofaBinPath: string;
}

export class SlurmError extends Error {
    constructor(message: string, public readonly stderrTail?: string) {
        super(message);
        this.name = 'SlurmError';
    }
}

// Banner parsers. All match the ANSI-stripped line so the regexes stay
// simple; see `stripAnsi` below.
const JOB_RE = /salloc:\s+Granted job allocation\s+(\d+)/;
// Matches the printed `ssh -N … -L <local>:<node>:<remote> <host>` line
// from src/ofa_server.py's Step 1 hint. Extracts node + REMOTE port.
const SSH_L_RE = /-L\s+\d+:(\S+?):(\d+)\s+\S+/;
const TOKEN_RE = /apiKey\s*=\s*(\S+)/;
const READY_RE = /\[ofa-serve\]\s+Ctrl\+C to stop\./;

/** 1-hour cap so a wildly wrong partition doesn't hang the extension forever. */
const READY_TIMEOUT_MS = 60 * 60 * 1000;

/** Number of trailing stderr lines to include in error messages. */
const STDERR_TAIL_LINES = 40;

/**
 * Kick off salloc → srun → ofa. Resolves once ofa is ready and we've
 * parsed the full endpoint tuple.
 */
export function connect(opts: SlurmOptions, logger: Logger): Promise<OfaEndpoint> {
    return new Promise<OfaEndpoint>((resolve, reject) => {
        // Delegate the whole allocation dance to bin/ofa. It already
        // does everything we would have to reimplement here: sacctmgr
        // account auto-detection, site.toml scheduler defaults,
        // /etc/profile sourcing, CUDA module loading, and the exact
        // salloc/srun/ofa exec chain we want. We just forward the
        // extension's ofa.slurm.* settings via the env vars bin/ofa
        // already respects (OFA_ACCOUNT / OFA_PARTITION / OFA_WALLTIME
        // / OFA_GRES), plus OFA_JOB_NAME=ofa-vscode so
        // adoptExistingAllocation() can find our jobs by name without
        // colliding with CLI-launched 'ofa' sessions.
        //
        // Run through `bash -l -c` so the login profile (e.g. things
        // sourced by /etc/profile.d/*) is available. We still call
        // ofa by its absolute path (opts.ofaBinPath) because most
        // users have `module load assistant` in an interactive
        // terminal, not their non-interactive login profile — so we
        // cannot rely on $PATH to find `ofa`. See resolveOfaBin()
        // below for how the absolute path is discovered.
        const innerParts = [shellQuote(opts.ofaBinPath), '--serve', '--serve-quiet'];
        if (opts.enableTools) innerParts.push('--serve-enable-tools');
        const inner = innerParts.join(' ');

        const env: NodeJS.ProcessEnv = { ...process.env, OFA_JOB_NAME: 'ofa-vscode' };
        if (opts.account) env.OFA_ACCOUNT = opts.account;
        if (opts.partition) env.OFA_PARTITION = opts.partition;
        if (opts.walltime) env.OFA_WALLTIME = opts.walltime;
        if (opts.gres) env.OFA_GRES = opts.gres;
        if (opts.model) env.OFA_MODEL = opts.model;

        logger.info(`spawn (via bash -lc): ${inner}`);
        logger.info(`env: OFA_JOB_NAME=${env.OFA_JOB_NAME} OFA_ACCOUNT=${env.OFA_ACCOUNT ?? '<auto>'} OFA_PARTITION=${env.OFA_PARTITION ?? '<site.toml>'} OFA_WALLTIME=${env.OFA_WALLTIME ?? '<site.toml>'} OFA_GRES=${env.OFA_GRES ?? '<site.toml>'} OFA_MODEL=${env.OFA_MODEL ?? '<bin/ofa default>'}`);

        const child = cp.spawn('bash', ['-l', '-c', inner], {
            stdio: ['ignore', 'pipe', 'pipe'],
            env
        });

        let jobId: string | null = null;
        let node: string | null = null;
        let port: number | null = null;
        let token: string | null = null;
        let settled = false;
        const stderrTail: string[] = [];

        const timeoutHandle = setTimeout(() => {
            if (settled) return;
            settled = true;
            child.kill('SIGTERM');
            reject(new SlurmError(
                `timed out waiting for ofa readiness after ${READY_TIMEOUT_MS / 1000}s`,
                stderrTail.join('')
            ));
        }, READY_TIMEOUT_MS);

        function pushTail(line: string): void {
            stderrTail.push(line + '\n');
            while (stderrTail.length > STDERR_TAIL_LINES) {
                stderrTail.shift();
            }
        }

        function tryResolve(): void {
            if (settled) return;
            if (jobId && node && port !== null && token) {
                settled = true;
                clearTimeout(timeoutHandle);
                resolve({
                    node,
                    port,
                    token,
                    baseUrl: `http://${node}:${port}/v1`,
                    jobId,
                    process: child
                });
            }
        }

        function handleLine(line: string, fromStderr: boolean): void {
            const clean = stripAnsi(line);
            if (fromStderr) pushTail(clean);
            logger.info(`ofa: ${clean}`);
            const jm = JOB_RE.exec(clean);
            if (jm) {
                jobId = jm[1];
                logger.info(`jobId=${jobId}`);
            }
            const sm = SSH_L_RE.exec(clean);
            if (sm) {
                node = sm[1];
                port = Number.parseInt(sm[2], 10);
                logger.info(`endpoint parsed: ${node}:${port}`);
            }
            const tm = TOKEN_RE.exec(clean);
            if (tm) {
                token = tm[1];
                logger.info(`token parsed (****${token.slice(-4)})`);
            }
            if (READY_RE.test(clean)) {
                logger.info('readiness sentinel seen');
                tryResolve();
                if (!settled) {
                    settled = true;
                    clearTimeout(timeoutHandle);
                    child.kill('SIGTERM');
                    reject(new SlurmError(
                        `saw ready sentinel but banner missing: jobId=${jobId} node=${node} port=${port} token=${token ? 'set' : 'null'}`,
                        stderrTail.join('')
                    ));
                }
            }
        }

        setupLineReader(child.stdout!, (l) => handleLine(l, false));
        setupLineReader(child.stderr!, (l) => handleLine(l, true));

        child.on('error', (err) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeoutHandle);
            reject(new SlurmError(`failed to spawn ofa (via bash -lc): ${err.message}. Is 'module load assistant' in your login shell rc?`));
        });
        child.on('exit', (code, signal) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeoutHandle);
            reject(new SlurmError(
                `ofa exited before serve was ready (code=${code} signal=${signal})`,
                stderrTail.join('')
            ));
        });
    });
}

/**
 * Release the SLURM allocation. Prefers a SIGTERM cascade through the
 * child process tree; falls back to explicit `scancel` if the child is
 * already gone (or if we adopted the endpoint and never had a child).
 */
export async function disconnect(endpoint: OfaEndpoint, logger: Logger): Promise<void> {
    const child = endpoint.process;
    if (child && !child.killed && child.exitCode === null) {
        logger.info(`SIGTERM to salloc/srun (pid=${child.pid})`);
        child.kill('SIGTERM');
        await new Promise<void>((resolvePromise) => {
            const t = setTimeout(() => {
                if (!child.killed && child.exitCode === null) {
                    logger.warn('graceful exit timed out; SIGKILL');
                    child.kill('SIGKILL');
                }
                resolvePromise();
            }, 5000);
            child.once('exit', () => {
                clearTimeout(t);
                resolvePromise();
            });
        });
    } else if (!child) {
        logger.info(`no child process (adopted endpoint); scancel-only teardown for job ${endpoint.jobId}`);
    }
    // Belt-and-braces scancel: SIGTERM cascade usually releases the
    // job, but srun's --pty can occasionally strand it, and adopted
    // endpoints never had a child in the first place. Ignoring
    // errors because "already released" is the normal case.
    try {
        logger.info(`scancel ${endpoint.jobId}`);
        await execFile('scancel', [endpoint.jobId], { timeout: 10_000 });
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.info(`scancel returned: ${msg} (likely already released)`);
    }
}

// -- helpers ---------------------------------------------------------------

// eslint-disable-next-line no-control-regex
const ANSI_RE = /\x1b\[[0-9;]*m/g;

function stripAnsi(s: string): string {
    return s.replace(ANSI_RE, '');
}

/**
 * Buffer a stream into complete lines and dispatch each. Handles
 * partial trailing lines correctly (kept in `buf` until the next \n).
 */
function setupLineReader(stream: NodeJS.ReadableStream, onLine: (line: string) => void): void {
    stream.setEncoding('utf-8');
    let buf = '';
    stream.on('data', (chunk: string | Buffer) => {
        buf += typeof chunk === 'string' ? chunk : chunk.toString();
        let idx: number;
        while ((idx = buf.indexOf('\n')) !== -1) {
            const line = buf.slice(0, idx).replace(/\r$/, '');
            buf = buf.slice(idx + 1);
            if (line.length > 0) onLine(line);
        }
    });
    stream.on('end', () => {
        if (buf.length > 0) onLine(buf);
    });
}

/**
 * Locate the `ofa` binary for this Kestrel install. Tried in order:
 *
 *   1. `ofa.ofaBinPath` setting if the user configured one explicitly.
 *   2. `$OFA_ROOT/bin/ofa` if `$OFA_ROOT` is exported in the
 *      extension host's env (e.g. via a system-wide profile.d
 *      script). Uncommon on Kestrel today but the cleanest signal
 *      when present.
 *   3. `bash -lic 'command -v ofa'` — an interactive login shell,
 *      which sources `~/.bashrc` and so picks up a `module load
 *      assistant` there. Note the `-i`: plain `-l` skips `.bashrc`,
 *      which is where most Kestrel users put their module loads.
 *   4. The Kestrel deploy path
 *      `/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/bin/ofa`.
 *      Last-ditch hard-coded fallback so the extension "just works"
 *      on a fresh Kestrel account without any prior module-load
 *      customisation.
 *
 * All candidates must be executable-by-us to count. Errors from each
 * step are swallowed and the search continues; only if all four fail
 * do we throw with an actionable message.
 */
export async function resolveOfaBin(configuredPath: string, logger: Logger): Promise<string> {
    const tryPath = async (p: string, source: string): Promise<string | null> => {
        try {
            await fs.access(p, fs.constants.X_OK);
            logger.info(`ofa binary: ${p} (source: ${source})`);
            return p;
        } catch {
            return null;
        }
    };

    // 1. Explicit setting
    const configured = configuredPath.trim();
    if (configured) {
        const found = await tryPath(configured, 'ofa.ofaBinPath setting');
        if (found) return found;
        logger.warn(`ofa.ofaBinPath='${configured}' is not an executable file; falling back to auto-detect`);
    }

    // 2. $OFA_ROOT env var
    const root = process.env.OFA_ROOT;
    if (root) {
        const found = await tryPath(path.join(root, 'bin', 'ofa'), '$OFA_ROOT/bin/ofa');
        if (found) return found;
    }

    // 3. Interactive login shell probe
    try {
        const { stdout } = await execFile(
            'bash', ['-lic', 'command -v ofa 2>/dev/null'],
            { timeout: 5000 }
        );
        const p = stdout.trim().split('\n')[0]?.trim();
        if (p) {
            const found = await tryPath(p, `bash -lic 'command -v ofa'`);
            if (found) return found;
        }
    } catch {
        // bash -i without a TTY can throw or exit non-zero even when
        // command -v prints. Swallow and continue.
    }

    // 4. Kestrel deploy default
    const kestrelPath = '/nopt/nrel/apps/cpu_stack/software/openfoam/assistant/bin/ofa';
    const found = await tryPath(kestrelPath, 'Kestrel deploy default');
    if (found) return found;

    throw new Error(
        `Could not locate 'ofa'. Tried ofa.ofaBinPath, $OFA_ROOT/bin/ofa, ` +
        `bash -lic 'command -v ofa', and ${kestrelPath}. ` +
        `Set 'ofa.ofaBinPath' in Settings to an absolute path.`
    );
}

/**
 * Minimal shell-quoting for the path we hand to `bash -l -c`. The
 * only realistic character to worry about in an install path is a
 * space; single-quoting handles that. Embedded single quotes are
 * escaped with the classic '"'"' idiom.
 */
function shellQuote(s: string): string {
    return "'" + s.replace(/'/g, "'\"'\"'") + "'";
}
