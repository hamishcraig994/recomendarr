import { describe, it, expect, vi } from 'vitest';
import { createChatCompletion } from '../src/lib/ai-recommender';
import type OpenAI from 'openai';

// Unit tests for the max_tokens / max_completion_tokens fallback, using a mocked client - the
// existing ai-recommender tests are live-integration style (real network calls, self-skipping
// without an API key), which can't deterministically exercise this specific retry path.

function fakeClient(create: (params: Record<string, unknown>) => Promise<unknown>) {
    return { chat: { completions: { create } } } as unknown as OpenAI;
}

describe('createChatCompletion', () => {
    it('uses max_tokens and does not retry when the model accepts it', async () => {
        const create = vi.fn().mockResolvedValue({ choices: [{ message: { content: 'ok' } }] });
        const client = fakeClient(create);

        await createChatCompletion(client, { model: 'gpt-4o', messages: [], max_tokens: 500 });

        expect(create).toHaveBeenCalledTimes(1);
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ max_tokens: 500 }));
        expect(create.mock.calls[0][0]).not.toHaveProperty('max_completion_tokens');
    });

    it('falls back to max_completion_tokens when the API rejects max_tokens for that reason', async () => {
        const rejection = new Error("Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.");
        const create = vi.fn()
            .mockRejectedValueOnce(rejection)
            .mockResolvedValueOnce({ choices: [{ message: { content: 'ok' } }] });
        const client = fakeClient(create);

        const result = await createChatCompletion(client, { model: 'o1-mini', messages: [], max_tokens: 500 });

        expect(create).toHaveBeenCalledTimes(2);
        expect(create.mock.calls[0][0]).toHaveProperty('max_tokens', 500);
        expect(create.mock.calls[1][0]).not.toHaveProperty('max_tokens');
        expect(create.mock.calls[1][0]).toHaveProperty('max_completion_tokens', 500);
        expect(result).toEqual({ choices: [{ message: { content: 'ok' } }] });
    });

    it('does not retry and rethrows on an unrelated error', async () => {
        const create = vi.fn().mockRejectedValue(new Error('401 Unauthorized'));
        const client = fakeClient(create);

        await expect(createChatCompletion(client, { model: 'gpt-4o', messages: [], max_tokens: 500 }))
            .rejects.toThrow('401 Unauthorized');
        expect(create).toHaveBeenCalledTimes(1);
    });

    it('preserves the other parameters (model, messages, temperature) on both attempts', async () => {
        const rejection = new Error("Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.");
        const create = vi.fn()
            .mockRejectedValueOnce(rejection)
            .mockResolvedValueOnce({ choices: [{ message: { content: 'ok' } }] });
        const client = fakeClient(create);

        const messages = [{ role: 'user' as const, content: 'hi' }];
        await createChatCompletion(client, { model: 'o3', messages, temperature: 0.7, max_tokens: 42 });

        for (const call of create.mock.calls) {
            expect(call[0]).toMatchObject({ model: 'o3', messages, temperature: 0.7 });
        }
    });
});
