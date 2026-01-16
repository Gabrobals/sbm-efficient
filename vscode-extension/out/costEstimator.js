"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.CostEstimator = void 0;
const vscode = __importStar(require("vscode"));
const MODEL_PRICING = {
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
class CostEstimator {
    config;
    totalTokensToday = 0;
    totalCostToday = 0;
    constructor() {
        this.config = vscode.workspace.getConfiguration('adaptive-k');
    }
    /**
     * Simple token counter (approximation: ~4 chars per token for English)
     * In production, use tiktoken for accurate counting
     */
    countTokens(text) {
        // Rough approximation: 1 token ≈ 4 characters
        // More accurate for code: 1 token ≈ 3.5 characters
        const charCount = text.length;
        return Math.ceil(charCount / 3.5);
    }
    /**
     * Get current model configuration
     */
    getModel() {
        const modelId = this.config.get('defaultModel', 'deepseek-v3');
        return MODEL_PRICING[modelId] || MODEL_PRICING['deepseek-v3'];
    }
    /**
     * Get configured savings rate
     */
    getSavingsRate() {
        return this.config.get('savingsRate', 35);
    }
    /**
     * Estimate cost for given token count
     */
    estimateCost(tokens, isOutput = false) {
        const model = this.getModel();
        const pricePerMillion = isOutput ? model.outputPrice : model.inputPrice;
        return (tokens / 1_000_000) * pricePerMillion;
    }
    /**
     * Calculate Adaptive-K savings
     */
    calculateSavings(baselineCost) {
        const savingsRate = this.getSavingsRate();
        return baselineCost * (savingsRate / 100);
    }
    /**
     * Get cost breakdown for a document
     */
    getDetailedEstimate(inputTokens, outputTokens = 0) {
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
    trackUsage(tokens, cost) {
        this.totalTokensToday += tokens;
        this.totalCostToday += cost;
    }
    /**
     * Get today's statistics
     */
    getTodayStats() {
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
    checkBudgetAlert() {
        const budget = this.config.get('dailyTokenBudget', 1_000_000);
        return this.totalTokensToday > budget * 0.8;
    }
    /**
     * Get all available models
     */
    static getAvailableModels() {
        return Object.values(MODEL_PRICING);
    }
    /**
     * Format currency
     */
    static formatCurrency(amount) {
        if (amount < 0.01) {
            return `$${amount.toFixed(6)}`;
        }
        else if (amount < 1) {
            return `$${amount.toFixed(4)}`;
        }
        else {
            return `$${amount.toFixed(2)}`;
        }
    }
}
exports.CostEstimator = CostEstimator;
//# sourceMappingURL=costEstimator.js.map