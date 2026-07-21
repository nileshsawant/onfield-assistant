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
    { id: 'ofa-vasp',              name: 'ofa · vasp',              tooltip: 'VASP (Vienna Ab initio Simulation Package) assistant.' },
    { id: 'ofa-reframe',           name: 'ofa · reframe',           tooltip: 'ReFrame CI/CD testing assistant.' }
];

/** Gemma 4's context is 32K tokens; keep the max-input just under to
 *  leave headroom for the RAG-augmented prompt ofa server adds. This
 *  matches the old chatLanguageModels.json byok setup users were on
 *  before this extension replaced it. */
const MAX_INPUT_TOKENS = 32000;
const MAX_OUTPUT_TOKENS = 8192;

const HTTP_TIMEOUT_MS = 5 * 60 * 1000;   // 5 min — long enough for a big generation

/**
 * Provider that always advertises the 8 ofa modes, whether or not
 * ofa is currently connected. A single instance is registered at
 * extension activation so Copilot Chat's picker sees our vendor
 * during its startup scan (registering later is unreliable — the
 * picker caches the vendor list). setEndpoint() wires up the live
 * ofa endpoint on connect and clears it on disconnect; while
 * cleared, provideLanguageModelChatResponse throws a helpful error
 * pointing the user at OFA: Connect.
 */
class OfaChatProvider implements vscode.LanguageModelChatProvider<vscode.LanguageModelChatInformation> {
    private endpoint: OfaEndpoint | null = null;
    private readonly _onDidChange = new vscode.EventEmitter<void>();
    readonly onDidChangeLanguageModelChatInformation: vscode.Event<void> = this._onDidChange.event;

    constructor(private readonly logger: Logger) {}

    /** Called from bringUp() after a successful connect, and from
     *  tearDown() with null on disconnect. Fires the change event so
     *  VS Code Chat re-queries our model list (which updates
     *  tooltips / details that mention the current node & job). */
    setEndpoint(endpoint: OfaEndpoint | null): void {
        this.endpoint = endpoint;
        this._onDidChange.fire();
    }

    dispose(): void {
        this._onDidChange.dispose();
    }

    async provideLanguageModelChatInformation(
        _options: vscode.PrepareLanguageModelChatModelOptions,
        _token: vscode.CancellationToken
    ): Promise<vscode.LanguageModelChatInformation[]> {
        const endpoint = this.endpoint;
        return OFA_MODES.map(({ id, name, tooltip }) => ({
            id,
            name,
            family: 'ofa',
            version: '1.0',
            vendor: 'ofa',
            tooltip: endpoint
                ? `${tooltip} (connected: ${endpoint.node}:${endpoint.port}, job ${endpoint.jobId})`
                : `${tooltip} (not connected — run 'OFA: Connect' from the command palette)`,
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
        const endpoint = this.endpoint;
        if (!endpoint) {
            const msg = `OFA is not connected. Run 'OFA: Connect' from the command palette to start a SLURM allocation, then retry.`;
            this.logger.warn(`chat request for ${model.id} rejected: not connected`);
            // Emit as a visible response part so the user sees the reason
            // in the chat pane, not just a generic error toast.
            progress.report(new vscode.LanguageModelTextPart(`⚠️ ${msg}`));
            return;
        }
        const url = `${endpoint.baseUrl}/chat/completions`;
        const openaiMessages = messages.map(toOpenAIMessage);
        const body = JSON.stringify({
            model: model.id,
            messages: openaiMessages,
            stream: true
        });

        this.logger.info(`chat request: model=${model.id} messages=${openaiMessages.length} endpoint=${endpoint.node}:${endpoint.port}`);

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
                    'authorization': `Bearer ${endpoint.token}`
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
 * Handle to the registered provider. Returned by registerOfaProvider()
 * at extension activation and passed back on connect/disconnect via
 * setEndpoint().
 */
export interface OfaProviderHandle {
    /** Attach or clear the live endpoint. Called from bringUp()
     *  (set) and tearDown() (clear). Fires an internal change event
     *  so VS Code Chat re-queries our model list — mostly a
     *  cosmetic refresh so the tooltip reflects the new node/job. */
    setEndpoint(endpoint: OfaEndpoint | null): void;
    /** Dispose the registration. Called from deactivate(). */
    dispose(): void;
}

/**
 * Register the ofa provider with VS Code at extension activation.
 * The provider always advertises the 8 ofa modes so they appear in
 * Copilot Chat's picker from startup — critical because the picker
 * caches its vendor list at first scan and doesn't re-scan reliably
 * when a provider registers later.
 *
 * The endpoint starts as null; a chat request in that state returns
 * a "run OFA: Connect first" message to the chat pane. When the
 * user connects, bringUp() calls handle.setEndpoint(endpoint) and
 * subsequent requests hit the real ofa server.
 */
export function registerOfaProvider(logger: Logger): OfaProviderHandle {
    const provider = new OfaChatProvider(logger);
    const registration = vscode.lm.registerLanguageModelChatProvider('ofa', provider);
    logger.info(`registered LanguageModelChatProvider (vendor=ofa, ${OFA_MODES.length} models, endpoint=<not connected>)`);
    return {
        setEndpoint(endpoint) {
            provider.setEndpoint(endpoint);
            logger.info(`provider endpoint updated: ${endpoint ? `${endpoint.node}:${endpoint.port}` : '<cleared>'}`);
        },
        dispose() {
            registration.dispose();
            provider.dispose();
            logger.info('LanguageModelChatProvider disposed');
        }
    };
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
