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
        const args = ['--serve', '--serve-quiet'];
        if (opts.enableTools) args.push('--serve-enable-tools');

        const env: NodeJS.ProcessEnv = { ...process.env, OFA_JOB_NAME: 'ofa-vscode' };
        if (opts.account) env.OFA_ACCOUNT = opts.account;
        if (opts.partition) env.OFA_PARTITION = opts.partition;
        if (opts.walltime) env.OFA_WALLTIME = opts.walltime;
        if (opts.gres) env.OFA_GRES = opts.gres;

        logger.info(`spawn: ofa ${args.join(' ')}`);
        logger.info(`env: OFA_JOB_NAME=${env.OFA_JOB_NAME} OFA_ACCOUNT=${env.OFA_ACCOUNT ?? '<auto>'} OFA_PARTITION=${env.OFA_PARTITION ?? '<site.toml>'} OFA_WALLTIME=${env.OFA_WALLTIME ?? '<site.toml>'} OFA_GRES=${env.OFA_GRES ?? '<site.toml>'}`);

        const child = cp.spawn('ofa', args, {
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
            reject(new SlurmError(`failed to spawn salloc: ${err.message}`));
        });
        child.on('exit', (code, signal) => {
            if (settled) return;
            settled = true;
            clearTimeout(timeoutHandle);
            reject(new SlurmError(
                `salloc/srun exited before ofa was ready (code=${code} signal=${signal})`,
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
