import * as vscode from 'vscode';
import { CostEstimator } from './costEstimator';

export class StatusBarManager implements vscode.Disposable {
    private statusBarItem: vscode.StatusBarItem;
    private costEstimator: CostEstimator;

    constructor(costEstimator: CostEstimator) {
        this.costEstimator = costEstimator;
        
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        
        this.statusBarItem.command = 'adaptive-k.estimateCost';
        this.statusBarItem.tooltip = 'Click for detailed cost analysis';
        this.statusBarItem.show();
    }

    update(document: vscode.TextDocument): void {
        const config = vscode.workspace.getConfiguration('adaptive-k');
        if (!config.get<boolean>('showStatusBar', true)) {
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
        this.statusBarItem.tooltip = new vscode.MarkdownString(
            `### Adaptive-K Cost Estimate\n\n` +
            `**Model:** ${model.name}\n\n` +
            `**Tokens:** ${tokens.toLocaleString()}\n\n` +
            `**Baseline:** ${CostEstimator.formatCurrency(cost)}\n\n` +
            `**With Adaptive-K:** ${CostEstimator.formatCurrency(adaptiveKCost)}\n\n` +
            `**Savings:** ${CostEstimator.formatCurrency(savings)} (${savingsPercent}%)\n\n` +
            `---\n\n` +
            `_Click for detailed analysis_`
        );

        // Color coding based on cost
        if (cost > 1) {
            this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
        } else {
            this.statusBarItem.backgroundColor = undefined;
        }

        this.statusBarItem.show();
    }

    private formatTokens(tokens: number): string {
        if (tokens >= 1_000_000) {
            return `${(tokens / 1_000_000).toFixed(1)}M`;
        } else if (tokens >= 1_000) {
            return `${(tokens / 1_000).toFixed(1)}K`;
        }
        return tokens.toString();
    }

    dispose(): void {
        this.statusBarItem.dispose();
    }
}
