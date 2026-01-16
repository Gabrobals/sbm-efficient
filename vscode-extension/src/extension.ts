import * as vscode from 'vscode';
import { CostEstimator } from './costEstimator';
import { StatusBarManager } from './statusBar';
import { DashboardPanel } from './dashboard';
import { MoEDetector } from './moeDetector';
import { ModelsTreeProvider, StatsTreeProvider } from './treeProviders';

export function activate(context: vscode.ExtensionContext) {
    console.log('Adaptive-K Toolkit is now active!');

    // Initialize components
    const costEstimator = new CostEstimator();
    const statusBar = new StatusBarManager(costEstimator);
    const moeDetector = new MoEDetector();

    // Register tree views
    const modelsProvider = new ModelsTreeProvider();
    const statsProvider = new StatsTreeProvider(costEstimator);
    
    vscode.window.registerTreeDataProvider('adaptive-k.models', modelsProvider);
    vscode.window.registerTreeDataProvider('adaptive-k.stats', statsProvider);

    // Register commands
    const estimateCostCmd = vscode.commands.registerCommand(
        'adaptive-k.estimateCost',
        async () => {
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
        }
    );

    const showDashboardCmd = vscode.commands.registerCommand(
        'adaptive-k.showDashboard',
        () => {
            DashboardPanel.createOrShow(context.extensionUri);
        }
    );

    const analyzeFileCmd = vscode.commands.registerCommand(
        'adaptive-k.analyzeFile',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showWarningMessage('No active editor');
                return;
            }

            const document = editor.document;
            const diagnostics = moeDetector.analyze(document);
            
            if (diagnostics.length === 0) {
                vscode.window.showInformationMessage('No MoE API calls detected in this file');
            } else {
                vscode.window.showInformationMessage(
                    `Found ${diagnostics.length} MoE API call(s). Adaptive-K could optimize these!`
                );
            }
        }
    );

    const insertSnippetCmd = vscode.commands.registerCommand(
        'adaptive-k.insertSnippet',
        async () => {
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
        }
    );

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
    context.subscriptions.push(
        estimateCostCmd,
        showDashboardCmd,
        analyzeFileCmd,
        insertSnippetCmd,
        onDidChangeActiveEditor,
        onDidChangeDocument,
        statusBar
    );
}

export function deactivate() {
    console.log('Adaptive-K Toolkit deactivated');
}
