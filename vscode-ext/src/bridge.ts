/**
 * TCP proxy bridge on the login node.
 *
 * ofa --serve binds inside the compute node's internal network. VS
 * Code Chat / Copilot Chat runs on the user's laptop and expects a
 * laptop-side localhost URL. Bridging is a two-hop:
 *
 *   laptop:<PORT>  --[VS Code Remote-SSH auto-forward]-->
 *      kl6:<PORT>  --[ncat listener + inner ncat per connection]-->
 *         <node>:<remote-port>  =  ofa
 *
 * The login-node listener is `ncat -l <PORT> --keep-open --sh-exec
 * "ncat <node> <remote-port>"`. Each incoming connection is forked
 * to a fresh `ncat` child that connects to the compute node — a
 * pure TCP relay, no ssh, no auth, no shell profile involvement.
 * Just plain nmap-ncat (already installed on Kestrel).
 *
 * <PORT> stays fixed across allocations, so the user's laptop-side
 * chatLanguageModels.json (URL http://localhost:<PORT> + the stable
 * bearer token from $OFA_SCRATCH/.ofa_api_key) stays valid forever —
 * only the compute-node hostname changes per allocation, and this
 * tunnel absorbs that.
 *
 * Why not ssh -L localhost? Kestrel login nodes don't run sshd on
 * 127.0.0.1 — `ssh localhost` fails with 'Connection refused'
 * regardless of ~/.ssh/authorized_keys. ncat sidesteps sshd
 * entirely.
 */
import * as cp from 'node:child_process';
import type { Logger } from './logger';
import type { OfaEndpoint } from './slurm';

/** How long to wait for ncat to either fail loudly OR settle into
 *  a working listener. ncat -l binds immediately or bails on
 *  address-in-use. Anything else (like ncat missing) fails at
 *  spawn time via the child.on('error') handler. */
const BRIDGE_STARTUP_MS = 1500;
const STDERR_TAIL_BYTES = 2000;

export interface BridgeHandle {
    /** Long-running ncat listener. Kill to tear down the tunnel. */
    process: cp.ChildProcess;
    /** Login-node port bound; same value the user's laptop-side
     *  chatLanguageModels.json points at. */
    localPort: number;
}

export function startBridge(
    endpoint: OfaEndpoint,
    localPort: number,
    logger: Logger
): Promise<BridgeHandle> {
    return new Promise<BridgeHandle>((resolve, reject) => {
        // Inner ncat is quoted so a whitespace-ful hostname (defensive;
        // Kestrel node names never contain spaces) doesn't break parsing.
        const innerCmd = `ncat ${endpoint.node} ${endpoint.port}`;
        const args = ['-l', String(localPort), '--keep-open', '--sh-exec', innerCmd];
        logger.info(`spawn: ncat ${args.join(' ')}`);

        const child = cp.spawn('ncat', args, {
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let settled = false;
        const stderrTail: string[] = [];
        let stderrTailBytes = 0;

        // ncat -l --keep-open is silent on success and stays alive.
        // Wait BRIDGE_STARTUP_MS: if the child hasn't exited or
        // errored by then, the listener is up and accepting.
        const readyTimer = setTimeout(() => {
            if (settled) return;
            settled = true;
            logger.info(`ncat bridge up: kl-node:${localPort} -> ${endpoint.node}:${endpoint.port}`);
            resolve({ process: child, localPort });
        }, BRIDGE_STARTUP_MS);

        child.stderr?.on('data', (chunk: Buffer | string) => {
            const s = chunk.toString();
            stderrTail.push(s);
            stderrTailBytes += s.length;
            while (stderrTailBytes > STDERR_TAIL_BYTES && stderrTail.length > 1) {
                stderrTailBytes -= stderrTail[0]!.length;
                stderrTail.shift();
            }
            for (const line of s.split('\n')) {
                if (line.trim()) logger.info(`ncat bridge: ${line}`);
            }
        });

        child.on('error', (err) => {
            if (settled) return;
            settled = true;
            clearTimeout(readyTimer);
            reject(new Error(`failed to spawn ncat: ${err.message}. Is ncat installed on the login node?`));
        });

        child.on('exit', (code, signal) => {
            if (settled) return;
            settled = true;
            clearTimeout(readyTimer);
            const tail = stderrTail.join('').trim();
            reject(new Error(
                `ncat bridge exited before it was ready (code=${code}, signal=${signal}).\n` +
                `Bridge command: ncat ${args.join(' ')}\n` +
                (tail ? `stderr tail:\n${tail}\n` : '') +
                `Common cause: port ${localPort} already bound on the login node ` +
                `(a previous OFA: Connect that didn't clean up, or another tool). ` +
                `Try running 'lsof -i :${localPort}' on the login node, or ` +
                `set 'ofa.laptopSideBridgePort' to a different free port.`
            ));
        });
    });
}

export async function stopBridge(handle: BridgeHandle, logger: Logger): Promise<void> {
    const child = handle.process;
    if (child.killed || child.exitCode !== null) return;
    logger.info(`stopping ncat bridge (pid=${child.pid}, port=${handle.localPort})`);
    child.kill('SIGTERM');
    await new Promise<void>((resolve) => {
        const t = setTimeout(() => {
            if (!child.killed && child.exitCode === null) {
                logger.warn('ncat SIGTERM timed out; SIGKILL');
                child.kill('SIGKILL');
            }
            resolve();
        }, 3000);
        child.once('exit', () => {
            clearTimeout(t);
            resolve();
        });
    });
}
