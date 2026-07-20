/**
 * OnField Assistant (ofa) — VS Code extension entry point.
 *
 * v0.1 case (a): Remote-SSH to Kestrel only. Auto-detects the Kestrel
 * host, spawns `salloc … srun … ofa --serve` as a child process on the
 * login node, parses the connection banner, and shows the endpoint in
 * the status bar. Model provider registration + streaming chat land in
 * PR 3; health probe + silent reconnect in PR 4; adopt-existing-
 * allocation + autoconnect in PR 5.
 *
 * See vscode-ext/README.md for the dev/build/debug loop.
 */
import * as vscode from 'vscode';
import { ChannelLogger, Logger } from './logger';
import { detectKestrel } from './kestrelDetector';
import { connect as slurmConnect, disconnect as slurmDisconnect, OfaEndpoint, SlurmError, SlurmOptions } from './slurm';

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
}

export async function deactivate(): Promise<void> {
    // VS Code awaits this (up to ~5s) before the extension host exits.
    // Best-effort scancel so we don't leak a SLURM allocation across
    // VS Code shutdowns.
    if (currentEndpoint && logger) {
        try {
            await slurmDisconnect(currentEndpoint, logger);
        } catch {
            // ignore — VS Code is shutting down anyway
        }
        currentEndpoint = null;
    }
}

async function connectCommand(): Promise<void> {
    if (!logger) return;
    if (currentEndpoint) {
        void vscode.window.showInformationMessage(
            `OFA already connected: ${currentEndpoint.node}:${currentEndpoint.port} (job ${currentEndpoint.jobId}). Run OFA: Disconnect first, or OFA: Re-allocate to swap.`
        );
        return;
    }
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

    const opts: SlurmOptions = {
        account: cfg.get<string>('slurm.account', ''),
        partition: cfg.get<string>('slurm.partition', 'debug'),
        walltime: cfg.get<string>('slurm.walltime', '00:30:00'),
        gres: cfg.get<string>('slurm.gres', 'gpu:1'),
        enableTools: cfg.get<boolean>('enableTools', true)
    };

    setStatus('connecting');
    try {
        currentEndpoint = await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: `OFA: allocating SLURM job (${opts.partition}, ${opts.walltime})…`,
                cancellable: false
            },
            () => slurmConnect(opts, logger!)
        );
        setStatus('connected');
        void vscode.window.showInformationMessage(
            `OFA connected: ${currentEndpoint.node}:${currentEndpoint.port} (job ${currentEndpoint.jobId}). Model picker registration lands in PR 3.`
        );
    } catch (err) {
        const isSlurm = err instanceof SlurmError;
        const msg = isSlurm ? err.message : (err instanceof Error ? err.message : String(err));
        logger.error(`connect failed: ${msg}`);
        if (isSlurm && err.stderrTail) {
            logger.error(`stderr tail:\n${err.stderrTail}`);
        }
        setStatus('disconnected');
        void vscode.window.showErrorMessage(`OFA: connect failed — ${msg}`, 'Show logs').then((choice) => {
            if (choice === 'Show logs') logChannel?.show(true);
        });
    }
}

async function disconnectCommand(): Promise<void> {
    if (!logger) return;
    if (!currentEndpoint) {
        void vscode.window.showInformationMessage('OFA is not connected.');
        return;
    }
    const endpoint = currentEndpoint;
    setStatus('disconnecting');
    try {
        await slurmDisconnect(endpoint, logger);
        logger.info(`disconnected: job ${endpoint.jobId} released`);
    } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(`disconnect error (allocation may already be released): ${msg}`);
    } finally {
        currentEndpoint = null;
        setStatus('disconnected');
    }
    void vscode.window.showInformationMessage('OFA disconnected; SLURM allocation released.');
}

async function reallocateCommand(): Promise<void> {
    if (currentEndpoint) await disconnectCommand();
    await connectCommand();
}

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
