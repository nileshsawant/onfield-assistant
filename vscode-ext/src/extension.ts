/**
 * OnField Assistant (ofa) — VS Code extension entry point.
 *
 * v0.1 case (a): Remote-SSH to Kestrel only. Auto-detects the Kestrel
 * host, spawns `salloc … srun … ofa --serve` on the login node,
 * parses the connection banner, registers a LanguageModelChatProvider
 * for the seven ofa modes, and health-probes the endpoint. When the
 * SLURM allocation expires, silently re-allocates and re-registers a
 * fresh provider — VS Code Chat's next request against the same
 * thread transparently lands on the new endpoint.
 *
 * See vscode-ext/README.md for the dev/build/debug loop.
 * PR 5 will add: adopt-existing-allocation on activate, autoconnect,
 * log-channel polish.
 */
import * as vscode from 'vscode';
import { ChannelLogger, Logger } from './logger';
import { detectKestrel } from './kestrelDetector';
import { connect as slurmConnect, disconnect as slurmDisconnect, resolveOfaBin, OfaEndpoint, SlurmError, SlurmOptions } from './slurm';
import { registerOfaProvider } from './modelProvider';
import { HealthProbe } from './healthProbe';
import { adoptExistingAllocation } from './adoptExisting';

const COMMAND_IDS = {
    connect: 'ofa.connect',
    disconnect: 'ofa.disconnect',
    reallocate: 'ofa.reallocate',
    showLogs: 'ofa.showLogs'
} as const;

let statusBarItem: vscode.StatusBarItem | undefined;
let logChannel: vscode.OutputChannel | undefined;
let logger: Logger | undefined;

let currentEndpoint: OfaEndpoint | null = null;
/** Disposable returned by vscode.lm.registerLanguageModelChatProvider —
 *  removes the seven ofa models from the picker when disposed. */
let providerRegistration: vscode.Disposable | null = null;
let healthProbe: HealthProbe | null = null;
/** True while we're mid-silent-reconnect. Prevents overlapping
 *  reallocations if the probe fires again while the first is running. */
let reconnecting = false;

export function activate(context: vscode.ExtensionContext): void {
    logChannel = vscode.window.createOutputChannel('OnField Assistant');
    context.subscriptions.push(logChannel);
    logger = new ChannelLogger(logChannel);
    logger.info(`activated (v${context.extension.packageJSON.version})`);
    logger.info(`extension host: ${vscode.env.remoteName ?? 'local'}`);

    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    setStatus('disconnected');
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    context.subscriptions.push(
        vscode.commands.registerCommand(COMMAND_IDS.connect, connectCommand),
        vscode.commands.registerCommand(COMMAND_IDS.disconnect, disconnectCommand),
        vscode.commands.registerCommand(COMMAND_IDS.reallocate, reallocateCommand),
        vscode.commands.registerCommand(COMMAND_IDS.showLogs, () => logChannel?.show(true))
    );

    // Post-activation, opportunistically adopt an existing allocation
    // from a previous session. If none exists (or the artifacts are
    // stale), maybeAutoConnect() runs the fresh-connect path IF the
    // user opted into ofa.autoConnectOnStartup. Both are best-effort
    // — we deliberately don't block activate() on them.
    void bootstrapConnection();
}

export async function deactivate(): Promise<void> {
    // VS Code awaits this (up to ~5s) before the extension host exits.
    // Best-effort scancel so we don't leak a SLURM allocation across
    // VS Code shutdowns.
    await tearDown({ silent: true });
}

// ---------------------------------------------------------------------------
// Bootstrap on activation: opportunistic adopt + optional autoconnect
// ---------------------------------------------------------------------------

/**
 * Runs shortly after activate(). Tries to silently adopt an existing
 * ofa-vscode SLURM allocation (avoids burning a queue wait on every
 * VS Code restart if the previous session left one running). If no
 * adoption target and ofa.autoConnectOnStartup is on, kicks off a
 * fresh silent connect. Otherwise stays quietly disconnected until
 * the user runs `OFA: Connect`.
 */
async function bootstrapConnection(): Promise<void> {
    if (!logger) return;
    const cfg = vscode.workspace.getConfiguration('ofa');

    // Adopt attempt is cheap (squeue + two file reads + one HTTP GET)
    // and non-Kestrel hosts are gated inside adoptExistingAllocation
    // by the required-env checks + squeue exec failing.
    if (vscode.env.remoteName === 'ssh-remote') {
        try {
            if (await tryAdopt({ silent: true })) return;
        } catch (err) {
            logger.info(`bootstrap adopt threw: ${(err as Error).message}`);
        }
    }

    if (cfg.get<boolean>('autoConnectOnStartup', false)) {
        logger.info('autoConnectOnStartup=true; running fresh connect');
        await bringUp({ silent: true });
    }
}

/**
 * Attempt to adopt an existing ofa-vscode SLURM job. Returns true iff
 * we successfully wired up currentEndpoint + provider + probe.
 */
async function tryAdopt(flow: FlowOptions): Promise<boolean> {
    if (!logger) return false;
    const adopted = await adoptExistingAllocation(logger);
    if (!adopted) return false;
    const cfg = vscode.workspace.getConfiguration('ofa');
    const healthIntervalSec = cfg.get<number>('healthProbeIntervalSeconds', 30);

    currentEndpoint = adopted;
    providerRegistration = registerOfaProvider(adopted, logger);
    healthProbe = new HealthProbe({
        baseUrl: adopted.baseUrl,
        token: adopted.token,
        intervalMs: Math.max(5, healthIntervalSec) * 1000,
        onDrop: () => void handleEndpointDrop(),
        logger
    });
    healthProbe.start();
    setStatus('connected');
    const notice = `OFA adopted existing allocation: ${adopted.node}:${adopted.port} (job ${adopted.jobId}).`;
    if (flow.silent) {
        logger.info(notice);
    } else {
        void vscode.window.showInformationMessage(notice);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async function connectCommand(): Promise<void> {
    if (!logger) return;
    if (currentEndpoint) {
        void vscode.window.showInformationMessage(
            `OFA already connected: ${currentEndpoint.node}:${currentEndpoint.port} (job ${currentEndpoint.jobId}). Run OFA: Disconnect first, or OFA: Re-allocate to swap.`
        );
        return;
    }
    // Prefer adoption over a fresh salloc so users don't burn a
    // queue wait if a prior allocation is still alive.
    if (await tryAdopt({ silent: false })) return;
    await bringUp({ silent: false });
}

async function disconnectCommand(): Promise<void> {
    if (!logger) return;
    if (!currentEndpoint) {
        void vscode.window.showInformationMessage('OFA is not connected.');
        return;
    }
    const jobId = currentEndpoint.jobId;
    await tearDown({ silent: false });
    void vscode.window.showInformationMessage(`OFA disconnected; SLURM job ${jobId} released.`);
}

async function reallocateCommand(): Promise<void> {
    if (currentEndpoint) await tearDown({ silent: true });
    await bringUp({ silent: false });
}

// ---------------------------------------------------------------------------
// Shared bring-up / tear-down (used by user commands AND the silent-
// reconnect flow triggered by the health probe).
// ---------------------------------------------------------------------------

interface FlowOptions {
    /** When true, suppress user-facing progress notifications. Errors
     *  are still surfaced. */
    silent: boolean;
}

async function bringUp(flow: FlowOptions): Promise<void> {
    if (!logger) return;
    const cfg = vscode.workspace.getConfiguration('ofa');
    const pattern = cfg.get<string>('kestrelHostnamePattern', '^(kl\\d+|kestrel|x\\d+c\\d+s\\d+b\\d+n\\d+)$');

    const probe = await detectKestrel(pattern);
    if (!probe.isKestrel) {
        logger.warn(`Kestrel probe failed: ${probe.reason}`);
        void vscode.window.showErrorMessage(
            `OFA: Connect refused — ${probe.reason}. This build only supports the Remote-SSH-to-Kestrel path (case a); case b lands in a later release.`
        );
        return;
    }
    logger.info(`Kestrel probe OK on '${probe.hostname}'`);

    let ofaBinPath: string;
    try {
        ofaBinPath = await resolveOfaBin(cfg.get<string>('ofaBinPath', ''), logger);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(msg);
        setStatus('disconnected');
        void vscode.window.showErrorMessage(`OFA: ${msg}`, 'Show logs').then((choice) => {
            if (choice === 'Show logs') logChannel?.show(true);
        });
        return;
    }

    const opts: SlurmOptions = {
        account: cfg.get<string>('slurm.account', ''),
        partition: cfg.get<string>('slurm.partition', 'debug'),
        walltime: cfg.get<string>('slurm.walltime', '00:30:00'),
        gres: cfg.get<string>('slurm.gres', 'gpu:1'),
        enableTools: cfg.get<boolean>('enableTools', true),
        ofaBinPath
    };
    const healthIntervalSec = cfg.get<number>('healthProbeIntervalSeconds', 30);

    setStatus('connecting');
    try {
        const runConnect = () => slurmConnect(opts, logger!);
        const endpoint = flow.silent
            ? await runConnect()
            : await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: `OFA: allocating SLURM job (${opts.partition}, ${opts.walltime})…`,
                    cancellable: false
                },
                runConnect
            );

        currentEndpoint = endpoint;
        providerRegistration = registerOfaProvider(endpoint, logger);
        healthProbe = new HealthProbe({
            baseUrl: endpoint.baseUrl,
            token: endpoint.token,
            intervalMs: Math.max(5, healthIntervalSec) * 1000,
            onDrop: () => void handleEndpointDrop(),
            logger
        });
        healthProbe.start();

        setStatus('connected');
        const notice = `OFA connected: ${endpoint.node}:${endpoint.port} (job ${endpoint.jobId}). Seven ofa models are now in the VS Code Chat model picker.`;
        if (flow.silent) {
            logger.info(notice);
        } else {
            void vscode.window.showInformationMessage(notice);
        }
    } catch (err) {
        const isSlurm = err instanceof SlurmError;
        const msg = isSlurm ? err.message : (err instanceof Error ? err.message : String(err));
        logger.error(`connect failed: ${msg}`);
        if (isSlurm && err.stderrTail) {
            logger.error(`stderr tail:\n${err.stderrTail}`);
        }
        setStatus('disconnected');
        // Even in silent mode, surface hard failures — the user's
        // chat is broken until they see this.
        void vscode.window.showErrorMessage(`OFA: connect failed — ${msg}`, 'Show logs').then((choice) => {
            if (choice === 'Show logs') logChannel?.show(true);
        });
    }
}

async function tearDown(flow: FlowOptions): Promise<void> {
    if (!logger) return;
    // Dispose the provider FIRST so no new chat request lands on a
    // stale endpoint mid-teardown. In-flight requests keep their
    // provider reference and get AbortController-cancelled by VS Code.
    if (providerRegistration) {
        providerRegistration.dispose();
        providerRegistration = null;
    }
    if (healthProbe) {
        healthProbe.stop();
        healthProbe = null;
    }
    if (currentEndpoint) {
        const endpoint = currentEndpoint;
        if (!flow.silent) setStatus('disconnecting');
        try {
            await slurmDisconnect(endpoint, logger);
            logger.info(`disconnected: job ${endpoint.jobId} released`);
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            logger.error(`disconnect error (allocation may already be released): ${msg}`);
        } finally {
            currentEndpoint = null;
        }
    }
    if (!flow.silent) setStatus('disconnected');
}

// ---------------------------------------------------------------------------
// Silent reconnect (triggered by the HealthProbe onDrop callback)
// ---------------------------------------------------------------------------

/**
 * The probe fired — endpoint appears dead. If the user opted for
 * silent reconnect (default), tear down and bring back up without a
 * modal; toast the outcome. Otherwise show a modal asking permission.
 */
async function handleEndpointDrop(): Promise<void> {
    if (!logger) return;
    if (reconnecting) {
        logger.info('handleEndpointDrop skipped: already reconnecting');
        return;
    }
    reconnecting = true;
    try {
        const cfg = vscode.workspace.getConfiguration('ofa');
        const silent = cfg.get<boolean>('silentReconnect', true);
        const previousJobId = currentEndpoint?.jobId;

        if (!silent) {
            const choice = await vscode.window.showWarningMessage(
                `OFA endpoint stopped responding (job ${previousJobId ?? '?'} likely expired). Re-allocate?`,
                { modal: false },
                'Re-allocate', 'Disconnect'
            );
            if (choice !== 'Re-allocate') {
                await tearDown({ silent: false });
                return;
            }
        } else {
            void vscode.window.showInformationMessage(
                `OFA endpoint dropped (job ${previousJobId ?? '?'}); re-allocating…`
            );
        }

        await tearDown({ silent: true });
        await bringUp({ silent: true });

        if (currentEndpoint) {
            const msg = `OFA reconnected (new job ${currentEndpoint.jobId}, node ${currentEndpoint.node}).`;
            logger.info(msg);
            void vscode.window.showInformationMessage(msg);
        }
    } finally {
        reconnecting = false;
    }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function setStatus(state: 'disconnected' | 'connecting' | 'connected' | 'disconnecting'): void {
    if (!statusBarItem) return;
    switch (state) {
        case 'disconnected':
            statusBarItem.text = '$(circle-slash) OFA: disconnected';
            statusBarItem.tooltip = 'Click to run OFA: Connect';
            statusBarItem.command = COMMAND_IDS.connect;
            break;
        case 'connecting':
            statusBarItem.text = '$(sync~spin) OFA: connecting…';
            statusBarItem.tooltip = 'Click to show OFA logs';
            statusBarItem.command = COMMAND_IDS.showLogs;
            break;
        case 'connected':
            if (currentEndpoint) {
                statusBarItem.text = `$(rocket) OFA: ${currentEndpoint.node}:${currentEndpoint.port}`;
                statusBarItem.tooltip = `Job ${currentEndpoint.jobId} — click to run OFA: Disconnect`;
            } else {
                statusBarItem.text = '$(rocket) OFA: connected';
                statusBarItem.tooltip = 'Click to run OFA: Disconnect';
            }
            statusBarItem.command = COMMAND_IDS.disconnect;
            break;
        case 'disconnecting':
            statusBarItem.text = '$(sync~spin) OFA: disconnecting…';
            statusBarItem.tooltip = 'Click to show OFA logs';
            statusBarItem.command = COMMAND_IDS.showLogs;
            break;
    }
}
