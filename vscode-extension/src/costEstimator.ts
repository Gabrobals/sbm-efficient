import * as vscode from 'vscode';

interface ModelPricing {
    name: string;
    inputPrice: number;  // per 1M tokens
    outputPrice: number; // per 1M tokens
    experts: number;
    topK: number;
    expectedSavings: number;
}

const MODEL_PRICING: Record<string, ModelPricing> = {
    'deepseek-v3': {
        name: 'DeepSeek-V3',
        inputPrice: 0.14,
        outputPrice: 0.28,
        experts: 256,
        topK: 8,
        expectedSavings: 35
    },
    'mixtral-8x7b': {
        name: 'Mixtral 8x7B',
        inputPrice: 0.24,
        outputPrice: 0.24,
        experts: 8,
        topK: 2,
        expectedSavings: 52.5
    },
    'qwen-moe': {
        name: 'Qwen1.5-MoE',
        inputPrice: 0.65,
        outputPrice: 0.65,
        experts: 60,
        topK: 4,
        expectedSavings: 32.4
    },
    'qwen3-235b': {
        name: 'Qwen3-235B-MoE',
        inputPrice: 0.65,
        outputPrice: 0.65,
        experts: 128,
        topK: 22,
        expectedSavings: 30
    },
    'olmoe-1b-7b': {
        name: 'OLMoE 1B-7B',
        inputPrice: 0.10,
        outputPrice: 0.10,
        experts: 64,
        topK: 8,
        expectedSavings: 24.7
    }
};

export class CostEstimator {
    private config: vscode.WorkspaceConfiguration;
    private totalTokensToday: number = 0;
    private totalCostToday: number = 0;

    constructor() {
        this.config = vscode.workspace.getConfiguration('adaptive-k');
    }

    /**
     * Simple token counter (approximation: ~4 chars per token for English)
     * In production, use tiktoken for accurate counting
     */
    countTokens(text: string): number {
        // Rough approximation: 1 token ≈ 4 characters
        // More accurate for code: 1 token ≈ 3.5 characters
        const charCount = text.length;
        return Math.ceil(charCount / 3.5);
    }

    /**
     * Get current model configuration
     */
    getModel(): ModelPricing {
        const modelId = this.config.get<string>('defaultModel', 'deepseek-v3');
        return MODEL_PRICING[modelId] || MODEL_PRICING['deepseek-v3'];
    }

    /**
     * Get configured savings rate
     */
    getSavingsRate(): number {
        return this.config.get<number>('savingsRate', 35);
    }

    /**
     * Estimate cost for given token count
     */
    estimateCost(tokens: number, isOutput: boolean = false): number {
        const model = this.getModel();
        const pricePerMillion = isOutput ? model.outputPrice : model.inputPrice;
        return (tokens / 1_000_000) * pricePerMillion;
    }

    /**
     * Calculate Adaptive-K savings
     */
    calculateSavings(baselineCost: number): number {
        const savingsRate = this.getSavingsRate();
        return baselineCost * (savingsRate / 100);
    }

    /**
     * Get cost breakdown for a document
     */
    getDetailedEstimate(inputTokens: number, outputTokens: number = 0): {
        inputCost: number;
        outputCost: number;
        totalCost: number;
        adaptiveKCost: number;
        savings: number;
        savingsPercent: number;
    } {
        const inputCost = this.estimateCost(inputTokens, false);
        const outputCost = this.estimateCost(outputTokens, true);
        const totalCost = inputCost + outputCost;
        const savings = this.calculateSavings(totalCost);
        
        return {
            inputCost,
            outputCost,
            totalCost,
            adaptiveKCost: totalCost - savings,
            savings,
            savingsPercent: this.getSavingsRate()
        };
    }

    /**
     * Track usage for daily statistics
     */
    trackUsage(tokens: number, cost: number): void {
        this.totalTokensToday += tokens;
        this.totalCostToday += cost;
    }

    /**
     * Get today's statistics
     */
    getTodayStats(): { tokens: number; cost: number; savings: number } {
        const savings = this.calculateSavings(this.totalCostToday);
        return {
            tokens: this.totalTokensToday,
            cost: this.totalCostToday,
            savings
        };
    }

    /**
     * Check if approaching daily budget
     */
    checkBudgetAlert(): boolean {
        const budget = this.config.get<number>('dailyTokenBudget', 1_000_000);
        return this.totalTokensToday > budget * 0.8;
    }

    /**
     * Get all available models
     */
    static getAvailableModels(): ModelPricing[] {
        return Object.values(MODEL_PRICING);
    }

    /**
     * Format currency
     */
    static formatCurrency(amount: number): string {
        if (amount < 0.01) {
            return `$${amount.toFixed(6)}`;
        } else if (amount < 1) {
            return `$${amount.toFixed(4)}`;
        } else {
            return `$${amount.toFixed(2)}`;
        }
    }
}
