/**
 * Adopt an existing ofa allocation instead of running a fresh salloc.
 *
 * If the previous VS Code session (or the CLI) left `ofa --serve`
 * running under a SLURM job named `ofa-vscode`, we can re-attach to
 * it on the next activate instead of burning a whole new queue wait +
 * ollama model reload. Cross-VS-Code-restart continuity.
 *
 * Detection sequence:
 *   1. `squeue -h -u $USER -n ofa-vscode -o '%N %i'` → node + jobId
 *      (single job per user by name — if multiple we take the first).
 *   2. Read $OFA_SCRATCH/.ofa_serve_port and $OFA_SCRATCH/.ofa_api_key
 *      to get port + token.
 *   3. GET http://<node>:<port>/healthz with Bearer <token> — if
 *      that returns 200 we adopt; otherwise the artifacts are stale
 *      and we bail (bringUp will run a fresh salloc).
 *
 * Deliberately restricted to the `ofa-vscode` job name so we don't
 * accidentally attach to (and later scancel) a CLI-launched ofa
 * that the user is actively using in a terminal.
 */
import * as cp from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { promisify } from 'node:util';
import type { Logger } from './logger';
import type { OfaEndpoint } from './slurm';

const execFile = promisify(cp.execFile);

const HEALTHZ_TIMEOUT_MS = 5000;
const SQUEUE_TIMEOUT_MS = 10_000;

export async function adoptExistingAllocation(logger: Logger): Promise<OfaEndpoint | null> {
    // 1. squeue by name.
    let node: string | null = null;
    let jobId: string | null = null;
    try {
        const { stdout } = await execFile(
            'squeue',
            ['-h', '-u', requireEnv('USER'), '-n', 'ofa-vscode', '-o', '%N %i'],
            { timeout: SQUEUE_TIMEOUT_MS }
        );
        const line = stdout.trim().split('\n')[0]?.trim();
        if (!line) {
            logger.info('adopt: no existing ofa-vscode SLURM job');
            return null;
        }
        const [nodeField, idField] = line.split(/\s+/);
        if (!nodeField || !idField) {
            logger.info(`adopt: unparseable squeue output '${line}'`);
            return null;
        }
        // squeue's %N can be a nodelist like 'kl3' or 'x3100c0s5b0n0'
        // or 'kl[3-5]'. For a 1-node ofa allocation it should be a
        // single hostname; if it's a range we bail rather than guess.
        if (nodeField.includes('[') || nodeField.includes(',')) {
            logger.warn(`adopt: unexpected multi-node allocation '${nodeField}'; skipping`);
            return null;
        }
        node = nodeField;
        jobId = idField;
        logger.info(`adopt: found candidate job ${jobId} on ${node}`);
    } catch (err) {
        logger.info(`adopt: squeue failed (${(err as Error).message}); skipping`);
        return null;
    }

    // 2. Read port + token from $OFA_SCRATCH artifacts.
    const scratch = process.env.OFA_SCRATCH ?? `/scratch/${requireEnv('USER')}`;
    const portPath = path.join(scratch, '.ofa_serve_port');
    const keyPath = path.join(scratch, '.ofa_api_key');
    let port: number;
    let token: string;
    try {
        const portRaw = (await fs.readFile(portPath, 'utf-8')).trim();
        port = Number.parseInt(portRaw, 10);
        if (!Number.isFinite(port) || port <= 0 || port > 65535) {
            logger.info(`adopt: garbage port in ${portPath}: '${portRaw}'`);
            return null;
        }
        token = (await fs.readFile(keyPath, 'utf-8')).trim();
        if (!token) {
            logger.info(`adopt: empty token in ${keyPath}`);
            return null;
        }
    } catch (err) {
        logger.info(`adopt: reading port/key files failed (${(err as Error).message}); skipping`);
        return null;
    }

    // 3. Probe /healthz to confirm the endpoint is alive AND the token
    // still authenticates. If the file artifacts got out of sync with
    // the running server (e.g. the CLI overwrote them) this catches it.
    const baseUrl = `http://${node}:${port}/v1`;
    const healthUrl = `http://${node}:${port}/healthz`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), HEALTHZ_TIMEOUT_MS);
    try {
        const res = await fetch(healthUrl, {
            method: 'GET',
            headers: { authorization: `Bearer ${token}` },
            signal: controller.signal
        });
        if (!res.ok) {
            logger.info(`adopt: /healthz returned HTTP ${res.status}; skipping`);
            return null;
        }
    } catch (err) {
        logger.info(`adopt: /healthz probe failed (${(err as Error).message}); skipping`);
        return null;
    } finally {
        clearTimeout(t);
    }

    logger.info(`adopt: reusing existing endpoint ${node}:${port} (job ${jobId})`);
    return {
        node: node!,
        port,
        token,
        baseUrl,
        jobId: jobId!,
        // We don't own the child, so on disconnect() we fall through
        // to `scancel <jobId>` alone (see slurm.ts disconnect()).
        process: null
    };
}

function requireEnv(name: string): string {
    const v = process.env[name];
    if (!v) throw new Error(`$${name} is not set in the extension host environment`);
    return v;
}
