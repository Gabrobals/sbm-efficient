import * as vscode from 'vscode';
import { CostEstimator } from './costEstimator';

export class ModelsTreeProvider implements vscode.TreeDataProvider<ModelItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<ModelItem | undefined | null | void> = 
        new vscode.EventEmitter<ModelItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<ModelItem | undefined | null | void> = 
        this._onDidChangeTreeData.event;

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: ModelItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ModelItem): Thenable<ModelItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        const models = CostEstimator.getAvailableModels();
        return Promise.resolve(
            models.map(m => new ModelItem(
                m.name,
                `${m.experts} experts, Top-${m.topK}`,
                `${m.expectedSavings}% savings`,
                vscode.TreeItemCollapsibleState.None
            ))
        );
    }
}

class ModelItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly description: string,
        public readonly tooltip: string,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState
    ) {
        super(label, collapsibleState);
        this.iconPath = new vscode.ThemeIcon('symbol-class');
    }
}

export class StatsTreeProvider implements vscode.TreeDataProvider<StatItem> {
    private _onDidChangeTreeData: vscode.EventEmitter<StatItem | undefined | null | void> = 
        new vscode.EventEmitter<StatItem | undefined | null | void>();
    readonly onDidChangeTreeData: vscode.Event<StatItem | undefined | null | void> = 
        this._onDidChangeTreeData.event;

    constructor(private costEstimator: CostEstimator) {}

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: StatItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: StatItem): Thenable<StatItem[]> {
        if (element) {
            return Promise.resolve([]);
        }

        const stats = this.costEstimator.getTodayStats();
        const model = this.costEstimator.getModel();

        return Promise.resolve([
            new StatItem('Current Model', model.name, 'robot'),
            new StatItem('Tokens Today', stats.tokens.toLocaleString(), 'symbol-number'),
            new StatItem('Cost Today', CostEstimator.formatCurrency(stats.cost), 'credit-card'),
            new StatItem('Savings Today', CostEstimator.formatCurrency(stats.savings), 'arrow-down'),
            new StatItem('Savings Rate', `${this.costEstimator.getSavingsRate()}%`, 'graph')
        ]);
    }
}

class StatItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly description: string,
        icon: string
    ) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}
