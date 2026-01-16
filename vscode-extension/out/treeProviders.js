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
exports.StatsTreeProvider = exports.ModelsTreeProvider = void 0;
const vscode = __importStar(require("vscode"));
const costEstimator_1 = require("./costEstimator");
class ModelsTreeProvider {
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve([]);
        }
        const models = costEstimator_1.CostEstimator.getAvailableModels();
        return Promise.resolve(models.map(m => new ModelItem(m.name, `${m.experts} experts, Top-${m.topK}`, `${m.expectedSavings}% savings`, vscode.TreeItemCollapsibleState.None)));
    }
}
exports.ModelsTreeProvider = ModelsTreeProvider;
class ModelItem extends vscode.TreeItem {
    label;
    description;
    tooltip;
    collapsibleState;
    constructor(label, description, tooltip, collapsibleState) {
        super(label, collapsibleState);
        this.label = label;
        this.description = description;
        this.tooltip = tooltip;
        this.collapsibleState = collapsibleState;
        this.iconPath = new vscode.ThemeIcon('symbol-class');
    }
}
class StatsTreeProvider {
    costEstimator;
    _onDidChangeTreeData = new vscode.EventEmitter();
    onDidChangeTreeData = this._onDidChangeTreeData.event;
    constructor(costEstimator) {
        this.costEstimator = costEstimator;
    }
    refresh() {
        this._onDidChangeTreeData.fire();
    }
    getTreeItem(element) {
        return element;
    }
    getChildren(element) {
        if (element) {
            return Promise.resolve([]);
        }
        const stats = this.costEstimator.getTodayStats();
        const model = this.costEstimator.getModel();
        return Promise.resolve([
            new StatItem('Current Model', model.name, 'robot'),
            new StatItem('Tokens Today', stats.tokens.toLocaleString(), 'symbol-number'),
            new StatItem('Cost Today', costEstimator_1.CostEstimator.formatCurrency(stats.cost), 'credit-card'),
            new StatItem('Savings Today', costEstimator_1.CostEstimator.formatCurrency(stats.savings), 'arrow-down'),
            new StatItem('Savings Rate', `${this.costEstimator.getSavingsRate()}%`, 'graph')
        ]);
    }
}
exports.StatsTreeProvider = StatsTreeProvider;
class StatItem extends vscode.TreeItem {
    label;
    description;
    constructor(label, description, icon) {
        super(label, vscode.TreeItemCollapsibleState.None);
        this.label = label;
        this.description = description;
        this.iconPath = new vscode.ThemeIcon(icon);
    }
}
//# sourceMappingURL=treeProviders.js.map