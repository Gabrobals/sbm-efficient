"""
VERTEX-RESEARCH n8n Integration

This module provides integration with n8n workflows for automated
research tracking following the VERTEX-RESEARCH protocol.

Usage:
    from src.utils.vertex_tracker import VertexTracker
    
    tracker = VertexTracker()
    tracker.start_experiment("EXP-001", config, git_commit)
    tracker.log_checkpoint(progress=50, seed=42, metrics={...})
    tracker.complete_experiment(results)
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. n8n integration disabled.")


@dataclass
class Hypothesis:
    """Registered hypothesis following Popperian principles."""
    id: str
    statement: str
    conditions: List[str]
    falsification_criteria: List[str]
    thresholds: Dict[str, float]
    registered_at: str
    status: str = "registered"


@dataclass
class ExperimentLog:
    """Experiment execution log."""
    experiment_id: str
    config: Dict[str, Any]
    git_commit: str
    started_at: str
    status: str = "running"
    checkpoints: List[Dict] = None
    results: Optional[Dict] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.checkpoints is None:
            self.checkpoints = []


class VertexTracker:
    """
    Integration with n8n VERTEX-RESEARCH automation.
    
    Tracks experiments, hypotheses, and gate reviews through n8n webhooks.
    Falls back to local JSON logging if n8n is unavailable.
    
    Environment Variables:
        N8N_WEBHOOK_BASE: Base URL for n8n webhooks
        VERTEX_LOCAL_LOG: Path for local fallback logs
    """
    
    def __init__(
        self,
        webhook_base: Optional[str] = None,
        local_log_path: Optional[str] = None,
        enabled: bool = True
    ):
        self.enabled = enabled and HAS_REQUESTS
        self.webhook_base = webhook_base or os.getenv(
            'N8N_WEBHOOK_BASE',
            'https://n8n.vertexdata.it/webhook'
        )
        self.local_log_path = Path(local_log_path or os.getenv(
            'VERTEX_LOCAL_LOG',
            'results/vertex_logs'
        ))
        self.local_log_path.mkdir(parents=True, exist_ok=True)
        
        # Current experiment state
        self.experiment_id: Optional[str] = None
        self.notion_page_id: Optional[str] = None
        self.current_log: Optional[ExperimentLog] = None
        
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def _send(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict]:
        """Send data to n8n webhook with fallback to local logging."""
        # Always log locally
        self._log_locally(endpoint, data)
        
        if not self.enabled:
            return None
            
        try:
            response = requests.post(
                f"{self.webhook_base}/{endpoint}",
                json=data,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            if response.ok:
                return response.json()
            else:
                print(f"n8n webhook failed: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print("n8n webhook timeout - logged locally")
            return None
        except Exception as e:
            print(f"n8n notification failed: {e} - logged locally")
            return None
    
    def _log_locally(self, endpoint: str, data: Dict[str, Any]) -> None:
        """Save log to local JSON file as fallback."""
        log_file = self.local_log_path / f"{endpoint}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
    
    def _generate_experiment_id(self, config: Dict) -> str:
        """Generate unique experiment ID."""
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
        return f"EXP-{datetime.now().strftime('%Y%m%d-%H%M')}-{config_hash}"
    
    # ==================== HYPOTHESIS MANAGEMENT ====================
    
    def register_hypothesis(
        self,
        statement: str,
        conditions: List[str],
        falsification_criteria: List[str],
        thresholds: Dict[str, float]
    ) -> Hypothesis:
        """
        Register a hypothesis before experiments (pre-registration).
        
        Args:
            statement: The hypothesis statement (must be falsifiable)
            conditions: List of conditions that must hold
            falsification_criteria: What would prove the hypothesis wrong
            thresholds: Quantitative success/failure thresholds
            
        Returns:
            Hypothesis object with registration details
        """
        hypothesis_id = f"H{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        hypothesis = Hypothesis(
            id=hypothesis_id,
            statement=statement,
            conditions=conditions,
            falsification_criteria=falsification_criteria,
            thresholds=thresholds,
            registered_at=datetime.now().isoformat()
        )
        
        response = self._send('vertex-hypothesis', {
            'hypothesis_id': hypothesis_id,
            'hypothesis': statement,
            'conditions': conditions,
            'falsification': falsification_criteria,
            'thresholds': thresholds,
            'timestamp': hypothesis.registered_at
        })
        
        print(f"✅ Hypothesis {hypothesis_id} registered")
        return hypothesis
    
    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
        results: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Evaluate hypothesis against experimental results.
        
        Args:
            hypothesis: The registered hypothesis
            results: Experimental results to check against thresholds
            
        Returns:
            Evaluation with status (corroborated/falsified/inconclusive)
        """
        evaluation = {
            'hypothesis_id': hypothesis.id,
            'status': 'corroborated',
            'checks': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for criterion in hypothesis.falsification_criteria:
            # Check each falsification criterion
            # This is simplified - in practice you'd parse the criterion
            check = {'criterion': criterion, 'triggered': False}
            evaluation['checks'].append(check)
        
        # Check thresholds
        for metric, threshold in hypothesis.thresholds.items():
            if metric in results:
                actual = results[metric]
                passed = actual >= threshold if 'min' in metric else actual <= threshold
                if not passed:
                    evaluation['status'] = 'falsified'
                evaluation['checks'].append({
                    'metric': metric,
                    'threshold': threshold,
                    'actual': actual,
                    'passed': passed
                })
        
        self._send('vertex-hypothesis-eval', evaluation)
        
        status_emoji = '✅' if evaluation['status'] == 'corroborated' else '❌'
        print(f"{status_emoji} Hypothesis {hypothesis.id}: {evaluation['status']}")
        
        return evaluation
    
    # ==================== EXPERIMENT TRACKING ====================
    
    def start_experiment(
        self,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None,
        git_commit: Optional[str] = None
    ) -> str:
        """
        Log experiment start.
        
        Args:
            config: Experiment configuration dict
            experiment_id: Optional custom ID (auto-generated if not provided)
            git_commit: Git commit hash (auto-detected if not provided)
            
        Returns:
            Experiment ID
        """
        self.experiment_id = experiment_id or self._generate_experiment_id(config)
        git_commit = git_commit or self._get_git_commit()
        
        self.current_log = ExperimentLog(
            experiment_id=self.experiment_id,
            config=config,
            git_commit=git_commit,
            started_at=datetime.now().isoformat()
        )
        
        response = self._send('vertex-experiment', {
            'event': 'start',
            'experiment_id': self.experiment_id,
            'config': config,
            'git_commit': git_commit,
            'timestamp': self.current_log.started_at
        })
        
        if response:
            self.notion_page_id = response.get('notion_page_id')
        
        print(f"🚀 Experiment {self.experiment_id} started")
        return self.experiment_id
    
    def log_checkpoint(
        self,
        progress: float,
        current_seed: int,
        metrics: Dict[str, float],
        notes: Optional[str] = None
    ) -> None:
        """
        Log experiment checkpoint (e.g., after each seed).
        
        Args:
            progress: Progress percentage (0-100)
            current_seed: Current random seed being processed
            metrics: Current metrics (e.g., avg_k, accuracy)
            notes: Optional notes about this checkpoint
        """
        checkpoint = {
            'progress': progress,
            'current_seed': current_seed,
            'metrics': metrics,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.current_log:
            self.current_log.checkpoints.append(checkpoint)
        
        self._send('vertex-experiment', {
            'event': 'checkpoint',
            'experiment_id': self.experiment_id,
            'notion_page_id': self.notion_page_id,
            'progress': progress,
            'current_seed': current_seed,
            'metrics': metrics,
            'notes': notes,
            'timestamp': checkpoint['timestamp']
        })
        
        print(f"📊 Checkpoint: {progress:.0f}% | Seed {current_seed} | {metrics}")
    
    def complete_experiment(
        self,
        results: Dict[str, Any],
        summary: Optional[str] = None
    ) -> None:
        """
        Log experiment completion with final results.
        
        Args:
            results: Final aggregated results
            summary: Optional summary text
        """
        if self.current_log:
            self.current_log.status = 'complete'
            self.current_log.results = results
            self.current_log.completed_at = datetime.now().isoformat()
            
            # Save complete log
            log_file = self.local_log_path / f"{self.experiment_id}.json"
            with open(log_file, 'w') as f:
                json.dump(asdict(self.current_log), f, indent=2)
        
        self._send('vertex-experiment', {
            'event': 'complete',
            'experiment_id': self.experiment_id,
            'notion_page_id': self.notion_page_id,
            'results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ Experiment {self.experiment_id} complete!")
        print(f"   Results: {json.dumps(results, indent=2)}")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None) -> None:
        """
        Log experiment error.
        
        Args:
            error_message: Description of the error
            exception: Optional exception object
        """
        if self.current_log:
            self.current_log.status = 'failed'
            self.current_log.error = error_message
            self.current_log.completed_at = datetime.now().isoformat()
            
            # Save error log
            log_file = self.local_log_path / f"{self.experiment_id}_ERROR.json"
            with open(log_file, 'w') as f:
                json.dump(asdict(self.current_log), f, indent=2)
        
        self._send('vertex-experiment', {
            'event': 'error',
            'experiment_id': self.experiment_id,
            'notion_page_id': self.notion_page_id,
            'error_message': error_message,
            'exception_type': type(exception).__name__ if exception else None,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"❌ Experiment {self.experiment_id} FAILED: {error_message}")
    
    # ==================== GATE REVIEWS ====================
    
    def request_gate_review(
        self,
        phase: int,
        artifacts: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Request automated gate review for phase transition.
        
        Args:
            phase: Current phase number (0-9)
            artifacts: Dict of artifacts to review (files, metrics, etc.)
            project_id: Optional project identifier
            
        Returns:
            Gate review result with passed/failed status
        """
        response = self._send('vertex-gate-review', {
            'phase': phase,
            'artifacts': artifacts,
            'project_id': project_id,
            'timestamp': datetime.now().isoformat()
        })
        
        if response and response.get('passed'):
            print(f"✅ Gate Review PASSED for Phase {phase}")
        else:
            print(f"⚠️ Gate Review requires attention for Phase {phase}")
            if response:
                print(f"   Missing: {response.get('missing', [])}")
        
        return response or {'passed': False, 'error': 'n8n unavailable'}
    
    # ==================== CONTEXT MANAGER ====================
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-complete or log error."""
        if exc_type is not None:
            self.log_error(str(exc_val), exc_val)
            return False  # Re-raise exception
        return False


# ==================== DECORATOR ====================

def track_experiment(config_fn=None, tracker: Optional[VertexTracker] = None):
    """
    Decorator to automatically track experiment functions.
    
    Usage:
        @track_experiment(lambda: {'model': 'mixtral', 'threshold': 1.275})
        def run_experiment():
            # ... experiment code ...
            return results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal tracker
            if tracker is None:
                tracker = VertexTracker()
            
            config = config_fn() if config_fn else {}
            tracker.start_experiment(config)
            
            try:
                results = func(*args, **kwargs)
                tracker.complete_experiment(results)
                return results
            except Exception as e:
                tracker.log_error(str(e), e)
                raise
        
        return wrapper
    return decorator


# ==================== CLI ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VERTEX-RESEARCH Tracker CLI')
    parser.add_argument('--test', action='store_true', help='Run test workflow')
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Testing VERTEX Tracker...\n")
        
        tracker = VertexTracker()
        
        # Test hypothesis registration
        h = tracker.register_hypothesis(
            statement="Entropy-based K selection reduces compute by >25%",
            conditions=["MoE architecture with learned router", "Calibrated thresholds"],
            falsification_criteria=["Compute savings < 15%", "Perplexity degradation > 2%"],
            thresholds={"min_savings": 0.25, "max_ppl_degradation": 0.02}
        )
        print()
        
        # Test experiment tracking
        tracker.start_experiment(
            config={"model": "mixtral", "k_values": [1, 2], "threshold": 1.275}
        )
        print()
        
        # Simulate checkpoints
        for i, seed in enumerate([42, 43, 44, 45, 46]):
            tracker.log_checkpoint(
                progress=(i + 1) * 20,
                current_seed=seed,
                metrics={"avg_k": 1.38 + (seed % 3) * 0.01, "ppl": 3.87}
            )
        print()
        
        # Complete
        results = {
            "avg_k_mean": 1.38,
            "avg_k_std": 0.02,
            "compute_savings": 0.31,
            "ppl_change": 0.008
        }
        tracker.complete_experiment(results)
        print()
        
        # Evaluate hypothesis
        tracker.evaluate_hypothesis(h, {"min_savings": 0.31, "max_ppl_degradation": 0.008})
        
        print("\n✅ Test complete! Check results/vertex_logs/ for local logs.")
