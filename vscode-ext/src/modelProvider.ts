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

/** Context window (num_ctx) and max generation length (num_predict)
 *  per OFA_MODEL id, mirroring MODEL_REGISTRY in src/ofa_main.py —
 *  keep these two in sync when the Python registry changes. The ''
 *  key is bin/ofa's own fallback when ofa.model is left empty
 *  (currently gemma4:31b-it-q8_0). */
const MODEL_CONTEXT: Readonly<Record<string, { numCtx: number; numPredict: number }>> = {
    '':                       { numCtx: 262144, numPredict: 32768 },
    'gemma4:31b':              { numCtx: 65536,  numPredict: 32768 },
    'gemma4:31b-it-q8_0':      { numCtx: 262144, numPredict: 32768 },
    'gemma4:26b':              { numCtx: 65536,  numPredict: 32768 },
    'llama4:scout':            { numCtx: 131072, numPredict: 16384 },
    'llama3.3:70b':            { numCtx: 32768,  numPredict: 16384 },
    'phi4:14b':                { numCtx: 16384,  numPredict: 8192 },
    'granite4:32b-a9b-h':      { numCtx: 131072, numPredict: 16384 },
    'gpt-oss:120b':            { numCtx: 65536,  numPredict: 32768 },
    'muse-glimmer:30b':        { numCtx: 131072, numPredict: 32768 }
};
/** Extra tokens to reserve on top of num_predict for the RAG-augmented
 *  prompt ofa server adds server-side (Copilot only sees the user's raw
 *  message here, not what ofa injects afterward). */
const RAG_HEADROOM_TOKENS = 4096;
const MIN_INPUT_TOKENS = 2048;

/** Compute the advertised input/output token limits for the model
 *  currently selected via the ofa.model setting, so Copilot Chat's
 *  client-side budget check (which runs before provideLanguageModelChatResponse
 *  is ever called) reflects the model's real context window instead of
 *  a stale fixed value — an overly conservative constant here silently
 *  rejects conversations the model could actually handle. */
function getModelLimits(): { maxInputTokens: number; maxOutputTokens: number } {
    const configured = vscode.workspace.getConfiguration('ofa').get<string>('model', '') ?? '';
    const entry = MODEL_CONTEXT[configured] ?? MODEL_CONTEXT[''];
    const maxOutputTokens = entry.numPredict;
    const maxInputTokens = Math.max(MIN_INPUT_TOKENS, entry.numCtx - entry.numPredict - RAG_HEADROOM_TOKENS);
    return { maxInputTokens, maxOutputTokens };
}

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
    private readonly configSub: vscode.Disposable;

    constructor(private readonly logger: Logger) {
        // ofa.model changes the underlying LLM's context window, and
        // ofa.enableTools changes the advertised toolCalling capability
        // while disconnected — both need to refresh without a reconnect
        // (setEndpoint() is the only other trigger for that).
        this.configSub = vscode.workspace.onDidChangeConfiguration((e) => {
            if (e.affectsConfiguration('ofa.model') || e.affectsConfiguration('ofa.enableTools')) {
                this._onDidChange.fire();
            }
        });
    }

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
        this.configSub.dispose();
    }

    async provideLanguageModelChatInformation(
        _options: vscode.PrepareLanguageModelChatModelOptions,
        _token: vscode.CancellationToken
    ): Promise<vscode.LanguageModelChatInformation[]> {
        const endpoint = this.endpoint;
        const { maxInputTokens, maxOutputTokens } = getModelLimits();
        // Reflects whichever ofa --serve is actually running (or, if not
        // connected yet, the setting the next connect would use) — the
        // server's --serve-enable-tools flag is fixed at spawn time and
        // can't change until a disconnect/reconnect.
        const toolCalling = endpoint
            ? endpoint.enableTools
            : vscode.workspace.getConfiguration('ofa').get<boolean>('enableTools', true);
        return OFA_MODES.map(({ id, name, tooltip }) => ({
            id,
            name,
            family: 'ofa',
            version: '1.0',
            vendor: 'ofa',
            tooltip: endpoint
                ? `${tooltip} (connected: ${endpoint.node}:${endpoint.port}, job ${endpoint.jobId})`
                : `${tooltip} (not connected — run 'OFA: Connect' from the command palette)`,
            maxInputTokens,
            maxOutputTokens,
            capabilities: {
                // ofa-serve translates OpenAI tool_calls to/from Ollama's
                // native tool-calling format when started with
                // --serve-enable-tools. See toOpenAIMessages()/streamSSE()
                // below for the wire-format translation.
                toolCalling,
                // Vision works when OFA_MODEL is vision-capable
                // (gemma4:31b, gemma4:31b-it-q8_0, gemma4:26b,
                // llama4:scout, muse-glimmer:30b). If OFA_MODEL is set
                // to a completion-only model (llama3.3:70b, phi4:14b,
                // granite4:32b-a9b-h) the ofa server will reject the
                // images. We advertise imageInput=true because the
                // deployment default gemma4:31b supports it; users who
                // pick a non-vision model via ofa.model implicitly opt
                // out of image uploads.
                imageInput: true
            }
        }));
    }

    async provideLanguageModelChatResponse(
        model: vscode.LanguageModelChatInformation,
        messages: readonly vscode.LanguageModelChatRequestMessage[],
        options: vscode.ProvideLanguageModelChatResponseOptions,
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
        const openaiMessages = messages.flatMap(toOpenAIMessages);
        const tools = toOpenAITools(options.tools);
        const body = JSON.stringify({
            model: model.id,
            messages: openaiMessages,
            stream: true,
            ...(tools ? { tools } : {})
        });

        this.logger.info(`chat request: model=${model.id} messages=${openaiMessages.length} tools=${tools?.length ?? 0} endpoint=${endpoint.node}:${endpoint.port}`);

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

/** Content in OpenAI's chat-completions API is either a plain string
 *  (text-only, cheapest to construct) or an array of typed parts for
 *  multimodal payloads. We use the array form only when at least one
 *  image is present so text-only messages stay compact on the wire. */
type OpenAIContentPart =
    | { type: 'text'; text: string }
    | { type: 'image_url'; image_url: { url: string } };

interface OpenAIToolCall {
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
}

interface OpenAIMessage {
    role: 'user' | 'assistant' | 'system' | 'tool';
    content: string | OpenAIContentPart[];
    tool_calls?: OpenAIToolCall[];
    tool_call_id?: string;
}

interface OpenAITool {
    type: 'function';
    function: { name: string; description?: string; parameters?: object };
}

/** Convert VS Code's declared tools (present only on Agent-mode
 *  requests) into the OpenAI `tools` array ofa-serve expects when
 *  started with --serve-enable-tools. Returns undefined when there
 *  are none, matching ofa_server.py treating an empty/missing list
 *  as "tools off" for this request. */
function toOpenAITools(tools: readonly vscode.LanguageModelChatTool[] | undefined): OpenAITool[] | undefined {
    if (!tools || tools.length === 0) return undefined;
    return tools.map(t => ({
        type: 'function',
        function: { name: t.name, description: t.description, parameters: t.inputSchema }
    }));
}

/**
 * Convert one VS Code chat message into one or more OpenAI messages.
 *
 * Assistant tool calls become `tool_calls` on an assistant message.
 * Tool results only ever appear inside a User-role message per the VS
 * Code API, but OpenAI (and ofa-serve, which matches results to calls
 * purely by `tool_call_id`) has no concept of a single message mixing
 * user text and tool results — so each LanguageModelToolResultPart
 * becomes its own `role: 'tool'` message, and any surrounding text
 * parts become separate `role: 'user'` messages.
 */
function toOpenAIMessages(msg: vscode.LanguageModelChatRequestMessage): OpenAIMessage[] {
    if (msg.role === vscode.LanguageModelChatMessageRole.Assistant) {
        const toolCalls: OpenAIToolCall[] = [];
        const rest: unknown[] = [];
        for (const part of msg.content) {
            if (part instanceof vscode.LanguageModelToolCallPart) {
                toolCalls.push({
                    id: part.callId,
                    type: 'function',
                    function: { name: part.name, arguments: JSON.stringify(part.input) }
                });
            } else {
                rest.push(part);
            }
        }
        const parts = partsToContentParts(rest);
        const out: OpenAIMessage = { role: 'assistant', content: contentPartsToWireForm(parts) };
        if (toolCalls.length > 0) out.tool_calls = toolCalls;
        return [out];
    }

    const out: OpenAIMessage[] = [];
    let buffer: unknown[] = [];
    const flush = () => {
        if (buffer.length === 0) return;
        out.push({ role: 'user', content: contentPartsToWireForm(partsToContentParts(buffer)) });
        buffer = [];
    };
    for (const part of msg.content) {
        if (part instanceof vscode.LanguageModelToolResultPart) {
            flush();
            out.push({ role: 'tool', tool_call_id: part.callId, content: toolResultToString(part.content) });
        } else {
            buffer.push(part);
        }
    }
    flush();
    return out.length > 0 ? out : [{ role: 'user', content: '' }];
}

/** Collapse content parts to a plain string when they're all text
 *  (the common case), keeping the array form only for multimodal
 *  (image) messages so text-only messages stay compact on the wire. */
function contentPartsToWireForm(parts: OpenAIContentPart[]): string | OpenAIContentPart[] {
    if (parts.length === 0) return '';
    if (parts.every(p => p.type === 'text')) {
        return parts.map(p => (p as { type: 'text'; text: string }).text).join('');
    }
    return parts;
}

/** Extract every part of a chat message into OpenAI content-part
 *  form. Unknown part types fall back to a bracketed marker so they
 *  appear in the conversation instead of vanishing silently. */
function partsToContentParts(parts: ReadonlyArray<unknown>): OpenAIContentPart[] {
    const out: OpenAIContentPart[] = [];
    for (const part of parts) {
        if (part instanceof vscode.LanguageModelTextPart) {
            out.push({ type: 'text', text: part.value });
        } else if (part instanceof vscode.LanguageModelToolCallPart) {
            out.push({ type: 'text', text: `[tool call: ${part.name}(${JSON.stringify(part.input)})]` });
        } else if (part instanceof vscode.LanguageModelToolResultPart) {
            out.push({ type: 'text', text: `[tool result for ${part.callId}: ${toolResultToString(part.content)}]` });
        } else {
            // LanguageModelDataPart (VS Code 1.98+, only surface for
            // image content today). Duck-type on {data, mimeType} so
            // the code compiles against the older @types/vscode
            // ^1.95 currently declared in package.json and remains
            // resilient to future part types with the same shape.
            const dp = part as unknown as { data?: Uint8Array; mimeType?: string };
            if (dp.data && dp.mimeType && dp.mimeType.startsWith('image/')) {
                const b64 = Buffer.from(dp.data).toString('base64');
                out.push({
                    type: 'image_url',
                    image_url: { url: `data:${dp.mimeType};base64,${b64}` }
                });
            } else {
                out.push({
                    type: 'text',
                    text: `[unsupported part: ${(part as { constructor?: { name?: string } })?.constructor?.name ?? typeof part}]`
                });
            }
        }
    }
    return out;
}

/** Stringify a tool result's content parts into the plain string
 *  ofa's `role: tool` messages expect — Ollama has no concept of
 *  structured tool-result content. */
function toolResultToString(content: ReadonlyArray<unknown>): string {
    const chunks: string[] = [];
    for (const part of content) {
        if (part instanceof vscode.LanguageModelTextPart) {
            chunks.push(part.value);
        } else {
            const withValue = part as { value?: unknown };
            chunks.push(typeof withValue.value === 'string' ? withValue.value : JSON.stringify(part));
        }
    }
    return chunks.join('\n');
}

/** Rough character-count string for token estimation only. Images
 *  are counted as a fixed budget matching Gemma 4's default
 *  ~280-visual-token cost so provideTokenCount() doesn't undercount
 *  multimodal messages. */
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
            const dp = part as unknown as { data?: Uint8Array; mimeType?: string };
            if (dp.data && dp.mimeType && dp.mimeType.startsWith('image/')) {
                // 280 tokens * 4 chars/token heuristic = 1120 chars.
                parts.push('X'.repeat(1120));
            } else {
                parts.push(`[unsupported part: ${(part as { constructor?: { name?: string } })?.constructor?.name ?? typeof part}]`);
            }
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
                            choices?: Array<{
                                delta?: {
                                    content?: string;
                                    tool_calls?: Array<{
                                        id?: string;
                                        function?: { name?: string; arguments?: string };
                                    }>;
                                };
                            }>;
                        };
                        const delta = parsed.choices?.[0]?.delta;
                        if (typeof delta?.content === 'string' && delta.content.length > 0) {
                            progress.report(new vscode.LanguageModelTextPart(delta.content));
                        }
                        // ofa-serve always sends one complete tool call per SSE
                        // chunk (Ollama returns tool_calls as a whole object
                        // rather than streaming arguments token-by-token like
                        // real OpenAI), so no cross-chunk accumulation is needed.
                        for (const tc of delta?.tool_calls ?? []) {
                            const name = tc.function?.name ?? '';
                            const argsRaw = tc.function?.arguments ?? '';
                            let input: object = {};
                            try {
                                input = argsRaw ? JSON.parse(argsRaw) : {};
                            } catch (err) {
                                logger.warn(`tool call arguments not valid JSON, using {}: ${(err as Error).message}`);
                            }
                            progress.report(new vscode.LanguageModelToolCallPart(tc.id ?? '', name, input));
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
