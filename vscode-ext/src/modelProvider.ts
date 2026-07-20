/**
 * LanguageModelChatProvider registration for the seven ofa modes.
 *
 * When we register this provider with vendor 'ofa', all seven models
 * appear in the VS Code Chat model picker automatically — no manual
 * chatLanguageModels.json editing required. When the user selects one
 * and asks a question, VS Code invokes
 * `provideLanguageModelChatResponse` on us with the full conversation
 * history; we POST it to `<baseUrl>/chat/completions` on the ofa
 * server and stream the SSE chunks back through the `progress` sink.
 *
 * This is the stable `vscode.lm.registerLanguageModelChatProvider`
 * API (as of VS Code 1.95). No --enable-proposed-api required.
 */
import * as vscode from 'vscode';
import type { Logger } from './logger';
import type { OfaEndpoint } from './slurm';

/**
 * The seven ofa modes advertised as separate models. `id` is what the
 * ofa server routes on (matches its /v1/models output); `name` is what
 * the user sees in the picker.
 */
const OFA_MODES: ReadonlyArray<{
    id: string;
    name: string;
    tooltip: string;
}> = [
    { id: 'ofa-code',              name: 'ofa · code',              tooltip: 'General coding assistant (default mode).' },
    { id: 'ofa-openfoam',          name: 'ofa · openfoam',          tooltip: 'OpenFOAM case + dictionary generator.' },
    { id: 'ofa-hpc',               name: 'ofa · hpc',               tooltip: 'HPC / Slurm / module system support.' },
    { id: 'ofa-amrex',             name: 'ofa · amrex',             tooltip: 'AMReX C++ framework assistant.' },
    { id: 'ofa-marbles',           name: 'ofa · marbles',           tooltip: 'MARBLES lattice-Boltzmann solver (on AMReX).' },
    { id: 'ofa-quantum-computing', name: 'ofa · quantum-computing', tooltip: 'Quantum computing assistant.' },
    { id: 'ofa-reframe',           name: 'ofa · reframe',           tooltip: 'ReFrame CI/CD testing assistant.' }
];

/** Gemma-4 31B has a ~8k context; keep some headroom for the reply. */
const MAX_INPUT_TOKENS = 7000;
const MAX_OUTPUT_TOKENS = 1024;

const HTTP_TIMEOUT_MS = 5 * 60 * 1000;   // 5 min — long enough for a big generation

/**
 * Provider that owns the ofa endpoint for the lifetime of a connection.
 * A fresh instance is created per `OFA: Connect` and disposed on
 * disconnect (so re-allocated jobs get a fresh provider with the new
 * node/port).
 */
class OfaChatProvider implements vscode.LanguageModelChatProvider<vscode.LanguageModelChatInformation> {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    onDidChangeLanguageModelChatInformation?: vscode.Event<void>;

    constructor(
        private readonly endpoint: OfaEndpoint,
        private readonly logger: Logger
    ) {}

    async provideLanguageModelChatInformation(
        _options: vscode.PrepareLanguageModelChatModelOptions,
        _token: vscode.CancellationToken
    ): Promise<vscode.LanguageModelChatInformation[]> {
        return OFA_MODES.map(({ id, name, tooltip }) => ({
            id,
            name,
            family: 'ofa',
            version: '1.0',
            vendor: 'ofa',
            tooltip,
            maxInputTokens: MAX_INPUT_TOKENS,
            maxOutputTokens: MAX_OUTPUT_TOKENS,
            capabilities: {
                // ofa --serve --serve-enable-tools translates OpenAI
                // tool_calls back and forth, but v0.1 of this extension
                // only forwards text. Flip when tool-call passthrough
                // lands in a later release.
                toolCalling: false,
                imageInput: false
            }
        }));
    }

    async provideLanguageModelChatResponse(
        model: vscode.LanguageModelChatInformation,
        messages: readonly vscode.LanguageModelChatRequestMessage[],
        _options: vscode.ProvideLanguageModelChatResponseOptions,
        progress: vscode.Progress<vscode.LanguageModelResponsePart>,
        token: vscode.CancellationToken
    ): Promise<void> {
        const url = `${this.endpoint.baseUrl}/chat/completions`;
        const openaiMessages = messages.map(toOpenAIMessage);
        const body = JSON.stringify({
            model: model.id,
            messages: openaiMessages,
            stream: true
        });

        this.logger.info(`chat request: model=${model.id} messages=${openaiMessages.length}`);

        // AbortController hooks the VS Code CancellationToken up to
        // fetch so 'Stop generating' actually stops the underlying
        // HTTP request rather than just discarding the buffer.
        const controller = new AbortController();
        const cancelSub = token.onCancellationRequested(() => {
            this.logger.info('chat request cancelled by user; aborting HTTP');
            controller.abort();
        });
        const timeoutId = setTimeout(() => controller.abort(), HTTP_TIMEOUT_MS);

        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: {
                    'content-type': 'application/json',
                    'authorization': `Bearer ${this.endpoint.token}`
                },
                body,
                signal: controller.signal
            });
            if (!res.ok) {
                const errText = await res.text().catch(() => '');
                throw new Error(`ofa returned HTTP ${res.status}: ${errText.slice(0, 500)}`);
            }
            if (!res.body) {
                throw new Error('ofa returned no response body');
            }
            await streamSSE(res.body, progress, this.logger);
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            if (controller.signal.aborted && !token.isCancellationRequested) {
                throw new Error(`ofa request timed out after ${HTTP_TIMEOUT_MS / 1000}s`);
            }
            if (token.isCancellationRequested) {
                // Silent — cancellation is normal.
                return;
            }
            this.logger.error(`chat request failed: ${msg}`);
            throw err;
        } finally {
            clearTimeout(timeoutId);
            cancelSub.dispose();
        }
    }

    async provideTokenCount(
        _model: vscode.LanguageModelChatInformation,
        text: string | vscode.LanguageModelChatRequestMessage,
        _token: vscode.CancellationToken
    ): Promise<number> {
        // Rough OpenAI heuristic: ~4 chars per token. Good enough for
        // VS Code Chat's budget accounting; ofa doesn't expose a real
        // tokenizer over HTTP.
        const s = typeof text === 'string' ? text : messageToString(text);
        return Math.ceil(s.length / 4);
    }
}

/**
 * Register the provider with VS Code. Returns a Disposable to hand
 * back to VS Code's subscription list — disposing it removes the
 * seven ofa models from the picker.
 */
export function registerOfaProvider(endpoint: OfaEndpoint, logger: Logger): vscode.Disposable {
    const provider = new OfaChatProvider(endpoint, logger);
    const disposable = vscode.lm.registerLanguageModelChatProvider('ofa', provider);
    logger.info(`registered LanguageModelChatProvider (vendor=ofa, ${OFA_MODES.length} models)`);
    return disposable;
}

// -- helpers ---------------------------------------------------------------

interface OpenAIMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
}

/**
 * Convert VS Code's LanguageModelChatRequestMessage into the OpenAI
 * chat-completions message shape. v0.1 handles text content only;
 * tool-call parts and data (image) parts are stringified into text so
 * they at least appear in the conversation instead of being dropped
 * silently — richer support lands with tool-call passthrough.
 */
function toOpenAIMessage(msg: vscode.LanguageModelChatRequestMessage): OpenAIMessage {
    const role: OpenAIMessage['role'] =
        msg.role === vscode.LanguageModelChatMessageRole.User ? 'user' : 'assistant';
    return { role, content: messageToString(msg) };
}

function messageToString(msg: vscode.LanguageModelChatRequestMessage): string {
    const parts: string[] = [];
    for (const part of msg.content) {
        if (part instanceof vscode.LanguageModelTextPart) {
            parts.push(part.value);
        } else if (part instanceof vscode.LanguageModelToolCallPart) {
            parts.push(`[tool call: ${part.name}(${JSON.stringify(part.input)})]`);
        } else if (part instanceof vscode.LanguageModelToolResultPart) {
            parts.push(`[tool result for ${part.callId}: ${JSON.stringify(part.content)}]`);
        } else {
            // LanguageModelDataPart or future variants — best-effort
            parts.push(`[unsupported part: ${(part as { constructor?: { name?: string } })?.constructor?.name ?? typeof part}]`);
        }
    }
    return parts.join('');
}

/**
 * Read an OpenAI-style SSE stream from `body`, parse each
 * `data: {…}` event, and forward `choices[0].delta.content` fragments
 * to the VS Code progress sink.
 *
 * The stream terminates on `data: [DONE]` or when the underlying
 * ReadableStream closes.
 */
async function streamSSE(
    body: ReadableStream<Uint8Array>,
    progress: vscode.Progress<vscode.LanguageModelResponsePart>,
    logger: Logger
): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let sawDone = false;
    try {
        // eslint-disable-next-line no-constant-condition
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            // SSE events are separated by a blank line (\n\n).
            let idx: number;
            while ((idx = buf.indexOf('\n\n')) !== -1) {
                const rawEvent = buf.slice(0, idx);
                buf = buf.slice(idx + 2);
                for (const line of rawEvent.split('\n')) {
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (payload === '[DONE]') {
                        sawDone = true;
                        continue;
                    }
                    if (payload === '') continue;
                    try {
                        const parsed = JSON.parse(payload) as {
                            choices?: Array<{ delta?: { content?: string } }>;
                        };
                        const delta = parsed.choices?.[0]?.delta?.content;
                        if (typeof delta === 'string' && delta.length > 0) {
                            progress.report(new vscode.LanguageModelTextPart(delta));
                        }
                    } catch (err) {
                        // Non-JSON data lines (heartbeats, comments) are
                        // legal in SSE; log at debug-ish level and keep
                        // reading rather than aborting the stream.
                        logger.warn(`SSE JSON parse skipped: ${(err as Error).message}`);
                    }
                }
            }
        }
        if (!sawDone) {
            logger.info('SSE stream closed without [DONE] marker (likely a network drop; response may be truncated)');
        }
    } finally {
        try {
            reader.releaseLock();
        } catch {
            // ignore
        }
    }
}
