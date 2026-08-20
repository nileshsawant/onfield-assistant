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
/** A failed bind is often transient (a just-killed previous bridge's
 *  socket not yet released by the OS, or similar) rather than a real
 *  conflict, so retry a few times with a short delay before giving up
 *  — bringUp()/tryAdopt() treat a bridge failure as non-fatal and
 *  never retry themselves, so without this a single transient failure
 *  left the bridge permanently down until a manual reconnect. */
const BRIDGE_RETRY_ATTEMPTS = 5;
const BRIDGE_RETRY_DELAY_MS = 750;

export interface BridgeHandle {
    /** Long-running ncat listener. Kill to tear down the tunnel. */
    process: cp.ChildProcess;
    /** Login-node port bound; same value the user's laptop-side
     *  chatLanguageModels.json points at. */
    localPort: number;
}

export async function startBridge(
    endpoint: OfaEndpoint,
    localPort: number,
    logger: Logger
): Promise<BridgeHandle> {
    let lastErr: Error | undefined;
    for (let attempt = 1; attempt <= BRIDGE_RETRY_ATTEMPTS; attempt++) {
        try {
            return await spawnBridgeOnce(endpoint, localPort, logger);
        } catch (err) {
            lastErr = err instanceof Error ? err : new Error(String(err));
            if (attempt < BRIDGE_RETRY_ATTEMPTS) {
                logger.warn(
                    `bridge start attempt ${attempt}/${BRIDGE_RETRY_ATTEMPTS} failed ` +
                    `(${lastErr.message.split('\n')[0]}); retrying in ${BRIDGE_RETRY_DELAY_MS}ms`
                );
                await new Promise<void>((r) => setTimeout(r, BRIDGE_RETRY_DELAY_MS));
            }
        }
    }
    throw lastErr;
}

function spawnBridgeOnce(
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
            stdio: ['ignore', 'pipe', 'pipe'],
            // --sh-exec forks a `sh -c "ncat <node> <port>"` child per
            // connection, which inherits a duplicate fd for the listening
            // socket. If that child hangs (e.g. relaying to a compute
            // node that just died mid-connection), killing only the
            // parent ncat leaves the listening port held open by the
            // orphaned grandchild. detached:true puts the whole tree in
            // its own process group so stopBridge() can kill all of it
            // at once via the negative pid.
            detached: true
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
    // Signal the whole process group (negative pid), not just the
    // tracked parent — per-connection --sh-exec children can outlive
    // it and keep the listening port held open otherwise (see the
    // detached:true comment in spawnBridgeOnce). Falls back to killing
    // just the parent if the group signal fails for any reason.
    const killGroup = (signal: NodeJS.Signals) => {
        try {
            if (child.pid) process.kill(-child.pid, signal);
            else child.kill(signal);
        } catch {
            try { child.kill(signal); } catch { /* already gone */ }
        }
    };
    await new Promise<void>((resolve) => {
        // Always resolve via the real 'exit' event, even after SIGKILL —
        // resolving early (before the OS actually reaps the process and
        // frees the port) used to race the next startBridge() call for
        // the same port, which then failed with "address already in
        // use" and left the bridge silently unbound after a reconnect.
        const hardKillTimer = setTimeout(() => {
            if (!child.killed && child.exitCode === null) {
                logger.warn('ncat SIGTERM timed out; SIGKILL (group)');
                killGroup('SIGKILL');
            }
        }, 3000);
        // Absolute last-resort cap so this can never hang the reconnect
        // flow forever; SIGKILL should make the OS reap the process
        // well before this fires.
        const giveUpTimer = setTimeout(() => {
            logger.warn(`ncat bridge (pid=${child.pid}) did not exit after SIGKILL; giving up waiting`);
            resolve();
        }, 8000);
        child.once('exit', () => {
            clearTimeout(hardKillTimer);
            clearTimeout(giveUpTimer);
            resolve();
        });
        killGroup('SIGTERM');
    });
}
