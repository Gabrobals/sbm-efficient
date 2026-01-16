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
exports.StatusBarManager = void 0;
const vscode = __importStar(require("vscode"));
const costEstimator_1 = require("./costEstimator");
class StatusBarManager {
    statusBarItem;
    costEstimator;
    constructor(costEstimator) {
        this.costEstimator = costEstimator;
        this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
        this.statusBarItem.command = 'adaptive-k.estimateCost';
        this.statusBarItem.tooltip = 'Click for detailed cost analysis';
        this.statusBarItem.show();
    }
    update(document) {
        const config = vscode.workspace.getConfiguration('adaptive-k');
        if (!config.get('showStatusBar', true)) {
            this.statusBarItem.hide();
            return;
        }
        const text = document.getText();
        const tokens = this.costEstimator.countTokens(text);
        const cost = this.costEstimator.estimateCost(tokens);
        const savings = this.costEstimator.calculateSavings(cost);
        const adaptiveKCost = cost - savings;
        // Format display
        const model = this.costEstimator.getModel();
        const savingsPercent = this.costEstimator.getSavingsRate();
        this.statusBarItem.text = `$(pulse) ${this.formatTokens(tokens)} | $(arrow-down) -${savingsPercent}%`;
        this.statusBarItem.tooltip = new vscode.MarkdownString(`### Adaptive-K Cost Estimate\n\n` +
            `**Model:** ${model.name}\n\n` +
            `**Tokens:** ${tokens.toLocaleString()}\n\n` +
            `**Baseline:** ${costEstimator_1.CostEstimator.formatCurrency(cost)}\n\n` +
            `**With Adaptive-K:** ${costEstimator_1.CostEstimator.formatCurrency(adaptiveKCost)}\n\n` +
            `**Savings:** ${costEstimator_1.CostEstimator.formatCurrency(savings)} (${savingsPercent}%)\n\n` +
            `---\n\n` +
            `_Click for detailed analysis_`);
        // Color coding based on cost
        if (cost > 1) {
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        }
        else {
            this.statusBarItem.backgroundColor = undefined;
        }
        this.statusBarItem.show();
    }
    formatTokens(tokens) {
        if (tokens >= 1_000_000) {
            return `${(tokens / 1_000_000).toFixed(1)}M`;
        }
        else if (tokens >= 1_000) {
            return `${(tokens / 1_000).toFixed(1)}K`;
        }
        return tokens.toString();
    }
    dispose() {
        this.statusBarItem.dispose();
    }
}
exports.StatusBarManager = StatusBarManager;
//# sourceMappingURL=statusBar.js.map