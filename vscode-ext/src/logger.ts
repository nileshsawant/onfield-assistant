/**
 * Thin wrapper around a VS Code OutputChannel that prefixes every line
 * with `[ofa-vscode]` so grepping the "OnField Assistant" output panel
 * feels the same as grepping the ofa CLI's stderr.
 */
import type * as vscode from 'vscode';

export interface Logger {
    info(message: string): void;
    warn(message: string): void;
    error(message: string): void;
}

export class ChannelLogger implements Logger {
    constructor(private readonly channel: vscode.OutputChannel) {}

    info(message: string): void {
        this.channel.appendLine(`[ofa-vscode] ${message}`);
    }
    warn(message: string): void {
        this.channel.appendLine(`[ofa-vscode] WARN: ${message}`);
    }
    error(message: string): void {
        this.channel.appendLine(`[ofa-vscode] ERROR: ${message}`);
    }
}
