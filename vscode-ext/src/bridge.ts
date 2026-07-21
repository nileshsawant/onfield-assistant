/**
 * SSH port-forward bridge on the login node.
 *
 * ofa --serve binds inside the compute node's internal network. VS
 * Code Chat / Copilot Chat runs on the user's laptop and expects a
 * laptop-side localhost URL. Bridging is a two-hop:
 *
 *   laptop:<PORT>  --[VS Code Remote-SSH auto-forward]-->
 *      kl6:<PORT>  --[ssh -N -L <PORT>:<node>:<remote-port>]-->
 *         <node>:<remote-port>  =  ofa
 *
 * We spawn `ssh -N -L <PORT>:<node>:<remote-port> localhost` on the
 * login node so kl6:<PORT> is a stable listener. <PORT> stays fixed
 * across allocations, so the user's laptop-side chatLanguageModels.json
 * (URL `http://localhost:<PORT>/v1/chat/completions` + the stable
 * bearer token from $OFA_SCRATCH/.ofa_api_key) stays valid forever —
 * only the compute-node hostname changes per allocation, and this
 * tunnel absorbs that.
 *
 * Requires: passwordless ssh to localhost on the login node (typical
 * on Kestrel because users have their own pubkey in authorized_keys).
 * On failure we surface the ssh stderr tail with the usual suspects
 * called out in the error message.
 */
import * as cp from 'node:child_process';
import type { Logger } from './logger';
import type { OfaEndpoint } from './slurm';

/** How long to wait for ssh -N -L to either fail loudly OR settle
 *  into a working tunnel. ssh -N -o BatchMode=yes prints stderr on
 *  auth failure / port-in-use within a few hundred ms, so 3 s is a
 *  comfortable margin. */
const BRIDGE_STARTUP_MS = 3000;
/** Trailing stderr bytes to preserve for error messages. */
const STDERR_TAIL_BYTES = 2000;

export interface BridgeHandle {
    /** Long-running ssh -N child. Kill to tear down the tunnel. */
    process: cp.ChildProcess;
    /** Login-node port bound by the tunnel; same value the user's
     *  laptop-side chatLanguageModels.json points at. */
    localPort: number;
}

export function startBridge(
    endpoint: OfaEndpoint,
    localPort: number,
    logger: Logger
): Promise<BridgeHandle> {
    return new Promise<BridgeHandle>((resolve, reject) => {
        const forwardSpec = `${localPort}:${endpoint.node}:${endpoint.port}`;
        const args = [
            '-N',
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ServerAliveInterval=60',
            '-L', forwardSpec,
            'localhost'
        ];
        logger.info(`spawn: ssh ${args.join(' ')}`);
        const child = cp.spawn('ssh', args, {
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let settled = false;
        const stderrTail: string[] = [];
        let stderrTailBytes = 0;

        // ssh -N in success emits nothing on stderr and stays alive.
        // We start a timer: if it hasn't exited or errored within
        // BRIDGE_STARTUP_MS, assume the tunnel is up.
        const readyTimer = setTimeout(() => {
            if (settled) return;
            settled = true;
            logger.info(`ssh -L up: kl-node:${localPort} -> ${endpoint.node}:${endpoint.port}`);
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
            // Log the raw stderr for diagnosis — passwordless-ssh
            // errors are the #1 cause of bridge failure.
            for (const line of s.split('\n')) {
                if (line.trim()) logger.info(`ssh -L: ${line}`);
            }
        });

        child.on('error', (err) => {
            if (settled) return;
            settled = true;
            clearTimeout(readyTimer);
            reject(new Error(`failed to spawn ssh -L: ${err.message}`));
        });

        child.on('exit', (code, signal) => {
            if (settled) return;
            settled = true;
            clearTimeout(readyTimer);
            const tail = stderrTail.join('').trim();
            reject(new Error(
                `ssh -L exited before tunnel was up (code=${code}, signal=${signal}).\n` +
                `Bridge command: ssh ${args.join(' ')}\n` +
                (tail ? `stderr tail:\n${tail}\n` : '') +
                `Common causes:\n` +
                `  • Passwordless ssh to localhost not set up ` +
                `(add your public key from ~/.ssh/id_ed25519.pub to ~/.ssh/authorized_keys).\n` +
                `  • Port ${localPort} already in use on the login node ` +
                `(a previous OFA: Connect that didn't clean up, or another tool).`
            ));
        });
    });
}

export async function stopBridge(handle: BridgeHandle, logger: Logger): Promise<void> {
    const child = handle.process;
    if (child.killed || child.exitCode !== null) return;
    logger.info(`stopping ssh -L (pid=${child.pid}, port=${handle.localPort})`);
    child.kill('SIGTERM');
    await new Promise<void>((resolve) => {
        const t = setTimeout(() => {
            if (!child.killed && child.exitCode === null) {
                logger.warn('ssh -L SIGTERM timed out; SIGKILL');
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
