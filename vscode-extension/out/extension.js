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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const costEstimator_1 = require("./costEstimator");
const statusBar_1 = require("./statusBar");
const dashboard_1 = require("./dashboard");
const moeDetector_1 = require("./moeDetector");
const treeProviders_1 = require("./treeProviders");
function activate(context) {
    console.log('Adaptive-K Toolkit is now active!');
    // Initialize components
    const costEstimator = new costEstimator_1.CostEstimator();
    const statusBar = new statusBar_1.StatusBarManager(costEstimator);
    const moeDetector = new moeDetector_1.MoEDetector();
    // Register tree views
    const modelsProvider = new treeProviders_1.ModelsTreeProvider();
    const statsProvider = new treeProviders_1.StatsTreeProvider(costEstimator);
    vscode.window.registerTreeDataProvider('adaptive-k.models', modelsProvider);
    vscode.window.registerTreeDataProvider('adaptive-k.stats', statsProvider);
    // Register commands
    const estimateCostCmd = vscode.commands.registerCommand('adaptive-k.estimateCost', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }
        const text = editor.document.getText();
        const tokens = costEstimator.countTokens(text);
        const cost = costEstimator.estimateCost(tokens);
        const savings = costEstimator.calculateSavings(cost);
        const message = `
📊 Token Count: ${tokens.toLocaleString()}
💰 Baseline Cost: $${cost.toFixed(4)}
✅ With Adaptive-K: $${(cost - savings).toFixed(4)}
💚 Savings: $${savings.toFixed(4)} (${costEstimator.getSavingsRate()}%)
            `.trim();
        vscode.window.showInformationMessage(message, 'Open Dashboard').then(selection => {
            if (selection === 'Open Dashboard') {
                vscode.commands.executeCommand('adaptive-k.showDashboard');
            }
        });
    });
    const showDashboardCmd = vscode.commands.registerCommand('adaptive-k.showDashboard', () => {
        dashboard_1.DashboardPanel.createOrShow(context.extensionUri);
    });
    const analyzeFileCmd = vscode.commands.registerCommand('adaptive-k.analyzeFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }
        const document = editor.document;
        const diagnostics = moeDetector.analyze(document);
        if (diagnostics.length === 0) {
            vscode.window.showInformationMessage('No MoE API calls detected in this file');
        }
        else {
            vscode.window.showInformationMessage(`Found ${diagnostics.length} MoE API call(s). Adaptive-K could optimize these!`);
        }
    });
    const insertSnippetCmd = vscode.commands.registerCommand('adaptive-k.insertSnippet', async () => {
        const snippets = [
            { label: 'Adaptive-K Basic Setup', value: 'adaptive-k-setup' },
            { label: 'Entropy-based Routing', value: 'entropy-routing' },
            { label: 'Cost Estimation', value: 'cost-estimation' },
            { label: 'Batch Processing', value: 'batch-processing' }
        ];
        const selected = await vscode.window.showQuickPick(snippets, {
            placeHolder: 'Select a snippet to insert'
        });
        if (selected) {
            vscode.commands.executeCommand('editor.action.insertSnippet', {
                name: selected.value
            });
        }
    });
    // Listen for document changes to update status bar
    const onDidChangeActiveEditor = vscode.window.onDidChangeActiveTextEditor(editor => {
        if (editor) {
            statusBar.update(editor.document);
        }
    });
    const onDidChangeDocument = vscode.workspace.onDidChangeTextDocument(event => {
        const editor = vscode.window.activeTextEditor;
        if (editor && event.document === editor.document) {
            statusBar.update(editor.document);
        }
    });
    // Initialize status bar for current editor
    if (vscode.window.activeTextEditor) {
        statusBar.update(vscode.window.activeTextEditor.document);
    }
    // Register disposables
    context.subscriptions.push(estimateCostCmd, showDashboardCmd, analyzeFileCmd, insertSnippetCmd, onDidChangeActiveEditor, onDidChangeDocument, statusBar);
}
function deactivate() {
    console.log('Adaptive-K Toolkit deactivated');
}
//# sourceMappingURL=extension.js.map