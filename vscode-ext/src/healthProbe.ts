/**
 * Periodic /healthz probe of the ofa endpoint.
 *
 * SLURM allocations expire on walltime. When that happens the compute
 * node terminates our ofa process, VS Code Chat's next
 * `provideLanguageModelChatResponse` gets ECONNREFUSED, and users see
 * a confusing error. This probe catches the drop before the user does
 * and lets extension.ts trigger the silent-reconnect flow (per the
 * design decision locked earlier: silent reconnect + toast).
 *
 * Semantics: the probe is a passive observer. It does NOT try to
 * reconnect itself — it just calls `onDrop()` once, then stops.
 * Restart it after a successful reconnect.
 */
import type { Logger } from './logger';

export interface HealthProbeOptions {
    /** Base URL of the ofa endpoint (as in OfaEndpoint.baseUrl). */
    baseUrl: string;
    /** Bearer token; sent as `Authorization: Bearer <token>` so
     *  auth-failed responses read as unhealthy. */
    token: string;
    /** Polling cadence. */
    intervalMs: number;
    /** Fired on the second consecutive failure so one blip doesn't
     *  trigger a full reallocation. */
    onDrop: () => void;
    logger: Logger;
}

/** Single failure could just be a network blip; require two in a row
 *  before declaring the endpoint dead. */
const FAILURE_THRESHOLD = 2;
/** Probe request timeout; keep short so a wedged server surfaces fast. */
const PROBE_TIMEOUT_MS = 5000;

export class HealthProbe {
    private timer: ReturnType<typeof setInterval> | null = null;
    private consecutiveFailures = 0;
    private fired = false;

    constructor(private readonly opts: HealthProbeOptions) {}

    /** Start polling. Idempotent — no-op if already running. */
    start(): void {
        if (this.timer) return;
        this.opts.logger.info(`healthz probe started (every ${this.opts.intervalMs / 1000}s)`);
        // Kick off an immediate probe so a dead endpoint at connect
        // time surfaces without waiting a full interval.
        void this.probe();
        this.timer = setInterval(() => void this.probe(), this.opts.intervalMs);
    }

    /** Stop polling. Safe to call multiple times. */
    stop(): void {
        if (!this.timer) return;
        clearInterval(this.timer);
        this.timer = null;
        this.opts.logger.info('healthz probe stopped');
    }

    private async probe(): Promise<void> {
        // Derive the /healthz URL from baseUrl (which ends in /v1).
        const url = this.opts.baseUrl.replace(/\/v1\/?$/, '/healthz');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
        try {
            const res = await fetch(url, {
                method: 'GET',
                headers: { 'authorization': `Bearer ${this.opts.token}` },
                signal: controller.signal
            });
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}`);
            }
            if (this.consecutiveFailures > 0) {
                this.opts.logger.info(`healthz recovered after ${this.consecutiveFailures} failure(s)`);
            }
            this.consecutiveFailures = 0;
        } catch (err) {
            this.consecutiveFailures++;
            const msg = err instanceof Error ? err.message : String(err);
            this.opts.logger.warn(`healthz probe ${this.consecutiveFailures}/${FAILURE_THRESHOLD} failed: ${msg}`);
            if (this.consecutiveFailures >= FAILURE_THRESHOLD && !this.fired) {
                this.fired = true;
                this.stop();
                this.opts.logger.warn('healthz threshold reached; firing onDrop');
                try {
                    this.opts.onDrop();
                } catch (dropErr) {
                    const dm = dropErr instanceof Error ? dropErr.message : String(dropErr);
                    this.opts.logger.error(`onDrop handler threw: ${dm}`);
                }
            }
        } finally {
            clearTimeout(timeoutId);
        }
    }
}
