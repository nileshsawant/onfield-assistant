/**
 * Refuses `OFA: Connect` on hosts that aren't a Kestrel login/compute
 * node. v0.1 case (a) requires the extension host to be running on
 * Kestrel via Remote-SSH; case (b) (local extension host reaching
 * Kestrel over ssh) lands in a later release.
 *
 * Detection is deliberately lenient — the hostname regex is
 * user-overridable via the `ofa.kestrelHostnamePattern` setting so
 * porters at peer HPC sites can reuse the extension without waiting
 * for us to grow a full site adapter.
 */
import * as vscode from 'vscode';
import * as cp from 'node:child_process';
import { promisify } from 'node:util';

const execFile = promisify(cp.execFile);

export interface KestrelProbeResult {
    /** True iff we're on a host matching `hostnamePattern`. */
    isKestrel: boolean;
    /** `uname -n` output. Empty if the probe never ran. */
    hostname: string;
    /** Human-readable rejection reason, or '' on success. */
    reason: string;
}

/**
 * Probe the current extension host to decide whether it looks like a
 * Kestrel node. Called from `OFA: Connect` before we spend time on
 * a SLURM allocation.
 */
export async function detectKestrel(hostnamePattern: string): Promise<KestrelProbeResult> {
    if (!vscode.env.remoteName) {
        return {
            isKestrel: false,
            hostname: '',
            reason: 'not running in a Remote-SSH session (extension host is local; this build only supports the Remote-SSH-to-Kestrel path)'
        };
    }
    if (vscode.env.remoteName !== 'ssh-remote') {
        return {
            isKestrel: false,
            hostname: '',
            reason: `extension host is on '${vscode.env.remoteName}', expected 'ssh-remote'`
        };
    }
    let hostname = '';
    try {
        const result = await execFile('uname', ['-n'], { timeout: 5000 });
        hostname = result.stdout.trim();
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return { isKestrel: false, hostname: '', reason: `uname -n failed: ${msg}` };
    }
    let re: RegExp;
    try {
        re = new RegExp(hostnamePattern);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
            isKestrel: false,
            hostname,
            reason: `invalid ofa.kestrelHostnamePattern regex (${msg})`
        };
    }
    if (!re.test(hostname)) {
        return {
            isKestrel: false,
            hostname,
            reason: `hostname '${hostname}' does not match pattern '${hostnamePattern}'`
        };
    }
    return { isKestrel: true, hostname, reason: '' };
}
