import { describe, expect, it } from "vitest";
import { getGemma4ThinkingLevel, isGemma4Model } from "../src/providers/google.js";
import { streamSimple } from "../src/stream.js";
import type { Context, Model, ThinkingLevel } from "../src/types.js";

type ClampedThinkingLevel = Exclude<ThinkingLevel, "xhigh">;

interface GoogleGenerateContentParameters {
	model: string;
	contents: unknown[];
	config?: {
		thinkingConfig?: {
			includeThoughts?: boolean;
			thinkingLevel?: string;
			thinkingBudget?: number;
		};
	};
}

function makeContext(): Context {
	return {
		messages: [{ role: "user", content: "Hello", timestamp: Date.now() }],
	};
}

async function captureThinkingConfig(
	model: Model<"google-generative-ai">,
	reasoning: ClampedThinkingLevel,
): Promise<{ thinkingLevel?: string; thinkingBudget?: number }> {
	let capturedConfig: GoogleGenerateContentParameters["config"] | undefined;

	const testModel: Model<"google-generative-ai"> = {
		...model,
		baseUrl: "http://127.0.0.1:9", // Invalid URL so request fails after payload capture
	};

	const s = streamSimple(testModel, makeContext(), {
		apiKey: "fake-key",
		reasoning,
		onPayload: (payload) => {
			capturedConfig = (payload as GoogleGenerateContentParameters).config;
			return payload;
		},
	});

	// The request will fail due to invalid URL, but onPayload fires before the network call
	await s.result();

	if (!capturedConfig?.thinkingConfig) {
		throw new Error("Expected thinkingConfig to be captured");
	}

	return capturedConfig.thinkingConfig;
}

describe("isGemma4Model", () => {
	it("detects gemma-4-31b-it", () => {
		expect(isGemma4Model({ id: "gemma-4-31b-it" } as Model<"google-generative-ai">)).toBe(true);
	});

	it("detects gemma-4-26b-a4b-it", () => {
		expect(isGemma4Model({ id: "gemma-4-26b-a4b-it" } as Model<"google-generative-ai">)).toBe(true);
	});

	it("detects gemma4 without hyphen", () => {
		expect(isGemma4Model({ id: "gemma4-31b-it" } as Model<"google-generative-ai">)).toBe(true);
	});

	it("rejects gemini models", () => {
		expect(isGemma4Model({ id: "gemini-2.5-pro" } as Model<"google-generative-ai">)).toBe(false);
		expect(isGemma4Model({ id: "gemini-3-pro" } as Model<"google-generative-ai">)).toBe(false);
		expect(isGemma4Model({ id: "gemini-3-flash" } as Model<"google-generative-ai">)).toBe(false);
	});

	it("rejects gemma non-4 models", () => {
		expect(isGemma4Model({ id: "gemma-3-12b" } as Model<"google-generative-ai">)).toBe(false);
		expect(isGemma4Model({ id: "gemma-2-27b" } as Model<"google-generative-ai">)).toBe(false);
	});

	it("is case-insensitive", () => {
		expect(isGemma4Model({ id: "GEMMA-4-31B-IT" } as Model<"google-generative-ai">)).toBe(true);
		expect(isGemma4Model({ id: "Gemma-4-31b-it" } as Model<"google-generative-ai">)).toBe(true);
	});
});

describe("getGemma4ThinkingLevel", () => {
	it("maps minimal to MINIMAL", () => {
		expect(getGemma4ThinkingLevel("minimal")).toBe("MINIMAL");
	});

	it("maps low to MINIMAL", () => {
		expect(getGemma4ThinkingLevel("low")).toBe("MINIMAL");
	});

	it("maps medium to HIGH", () => {
		expect(getGemma4ThinkingLevel("medium")).toBe("HIGH");
	});

	it("maps high to HIGH", () => {
		expect(getGemma4ThinkingLevel("high")).toBe("HIGH");
	});
});

describe("Gemma 4 thinking routing in streamSimple", () => {
	const gemma4Model: Model<"google-generative-ai"> = {
		id: "gemma-4-31b-it",
		name: "Gemma 4 31B",
		provider: "google-generative-ai",
		api: "google-generative-ai",
		reasoning: true,
		input: ["text"],
		contextWindow: 128000,
		maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		baseUrl: "",
	};

	const gemma4AltModel: Model<"google-generative-ai"> = {
		id: "gemma-4-26b-a4b-it",
		name: "Gemma 4 26B A4B",
		provider: "google-generative-ai",
		api: "google-generative-ai",
		reasoning: true,
		input: ["text"],
		contextWindow: 128000,
		maxTokens: 8192,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		baseUrl: "",
	};

	it("uses thinkingLevel (not thinkingBudget) for gemma-4-31b-it with high reasoning", async () => {
		const thinkingConfig = await captureThinkingConfig(gemma4Model, "high");
		expect(thinkingConfig.thinkingLevel).toBe("HIGH");
		expect(thinkingConfig.thinkingBudget).toBeUndefined();
	});

	it("uses thinkingLevel (not thinkingBudget) for gemma-4-26b-a4b-it with medium reasoning", async () => {
		const thinkingConfig = await captureThinkingConfig(gemma4AltModel, "medium");
		expect(thinkingConfig.thinkingLevel).toBe("HIGH");
		expect(thinkingConfig.thinkingBudget).toBeUndefined();
	});

	it("maps low reasoning to MINIMAL thinkingLevel for gemma-4", async () => {
		const thinkingConfig = await captureThinkingConfig(gemma4Model, "low");
		expect(thinkingConfig.thinkingLevel).toBe("MINIMAL");
		expect(thinkingConfig.thinkingBudget).toBeUndefined();
	});

	it("maps minimal reasoning to MINIMAL thinkingLevel for gemma-4", async () => {
		const thinkingConfig = await captureThinkingConfig(gemma4Model, "minimal");
		expect(thinkingConfig.thinkingLevel).toBe("MINIMAL");
		expect(thinkingConfig.thinkingBudget).toBeUndefined();
	});
});
