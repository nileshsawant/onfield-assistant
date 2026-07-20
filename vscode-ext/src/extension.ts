/**
 * OnField Assistant (ofa) — VS Code extension entry point.
 *
 * v0.1 SKELETON: registers the four command palette entries + a status
 * bar item, but the commands themselves are stubs that show a "not yet
 * implemented" notice. Real behaviour lands in subsequent commits:
 *
 *   PR 2 — kestrelDetector.ts + slurm.ts (banner parser, salloc/srun
 *          runner, JOBID tracking, scancel on disconnect).
 *   PR 3 — modelProvider.ts (register a LanguageModelChatProvider that
 *          streams responses from the ofa HTTP endpoint into VS Code
 *          Chat's model picker).
 *   PR 4 — healthProbe.ts (poll /healthz; silent reconnect on drop).
 *   PR 5 — polish: adopt-existing-allocation flow, autoconnect setting,
 *          logging channel improvements.
 *
 * See vscode-ext/README.md for the dev/build/debug loop.
 */
import * as vscode from 'vscode';

const COMMAND_IDS = {
    connect: 'ofa.connect',
    disconnect: 'ofa.disconnect',
    reallocate: 'ofa.reallocate',
    showLogs: 'ofa.showLogs'
} as const;

let statusBarItem: vscode.StatusBarItem | undefined;
let logChannel: vscode.OutputChannel | undefined;

export function activate(context: vscode.ExtensionContext): void {
    logChannel = vscode.window.createOutputChannel('OnField Assistant');
    context.subscriptions.push(logChannel);
    logChannel.appendLine(`[ofa-vscode] activated (v${context.extension.packageJSON.version})`);
    logChannel.appendLine(`[ofa-vscode] extension host: ${vscode.env.remoteName ?? 'local'}`);

    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = '$(circle-slash) OFA: disconnected';
    statusBarItem.tooltip = 'Click to run OFA: Connect';
    statusBarItem.command = COMMAND_IDS.connect;
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    context.subscriptions.push(
        vscode.commands.registerCommand(COMMAND_IDS.connect, () => notImplemented('Connect')),
        vscode.commands.registerCommand(COMMAND_IDS.disconnect, () => notImplemented('Disconnect')),
        vscode.commands.registerCommand(COMMAND_IDS.reallocate, () => notImplemented('Re-allocate SLURM job')),
        vscode.commands.registerCommand(COMMAND_IDS.showLogs, () => logChannel?.show(true))
    );
}

export function deactivate(): void {
    logChannel?.appendLine('[ofa-vscode] deactivated');
    // context.subscriptions already handled by VS Code; nothing to clean up
    // in the skeleton. Real teardown (scancel, tunnel close, provider
    // dispose) lands in PR 4.
}

/**
 * Placeholder used by every command in the skeleton so users get a
 * meaningful message instead of a silent no-op. Replaced with real
 * implementations in the follow-up PRs listed at the top of this file.
 */
function notImplemented(commandLabel: string): void {
    const message = `OFA: ${commandLabel} — not yet implemented in this skeleton build.`;
    logChannel?.appendLine(`[ofa-vscode] ${message}`);
    void vscode.window.showInformationMessage(message, 'Open logs').then((choice) => {
        if (choice === 'Open logs') {
            logChannel?.show(true);
        }
    });
}
