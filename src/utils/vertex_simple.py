"""
VERTEX-RESEARCH Simple Tracker

Zero-cost automation using:
- Local JSON files (always works)
- Google Sheets (free database)
- Gmail (free notifications)
- GitHub Actions (free automation)

Usage:
    python -m src.utils.vertex_simple project new "My Research"
    python -m src.utils.vertex_simple hypothesis register --statement "..."
    python -m src.utils.vertex_simple experiment start --config '{...}'
    python -m src.utils.vertex_simple daily --notes "..."
    python -m src.utils.vertex_simple report weekly
"""

import os
import json
import hashlib
import smtplib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

# Try to load optional dependencies
try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==================== CONFIGURATION ====================

class Config:
    """Configuration from environment variables."""
    
    SHEET_ID = os.getenv('VERTEX_SHEET_ID', '')
    GMAIL_USER = os.getenv('GMAIL_USER', '')
    GMAIL_APP_PASSWORD = os.getenv('GMAIL_APP_PASSWORD', '')
    RESEARCHER_NAME = os.getenv('RESEARCHER_NAME', 'Researcher')
    
    # Local paths
    BASE_DIR = Path(__file__).parent.parent.parent
    VERTEX_DIR = BASE_DIR / 'results' / 'vertex'
    PROJECTS_DIR = VERTEX_DIR / 'projects'
    HYPOTHESES_DIR = VERTEX_DIR / 'hypotheses'
    EXPERIMENTS_DIR = VERTEX_DIR / 'experiments'
    DAILY_DIR = VERTEX_DIR / 'daily'
    
    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        for d in [cls.PROJECTS_DIR, cls.HYPOTHESES_DIR, cls.EXPERIMENTS_DIR, cls.DAILY_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ==================== DATA CLASSES ====================

@dataclass
class Project:
    id: str
    name: str
    phase: int = 0
    status: str = "active"
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    
    PHASES = [
        "Problem Identification",
        "Literature Review", 
        "Theoretical Framework",
        "Hypothesis Formulation",
        "Experimental Design",
        "Implementation",
        "Experimentation",
        "Validation/Falsification",
        "Analysis & Interpretation",
        "Paper Writing",
        "Publication"
    ]
    
    @property
    def phase_name(self) -> str:
        return f"Phase {self.phase}: {self.PHASES[self.phase]}"


@dataclass 
class Hypothesis:
    id: str
    project_id: str
    statement: str
    conditions: List[str]
    falsification_criteria: List[str]
    thresholds: Dict[str, float] = field(default_factory=dict)
    status: str = "registered"
    registered: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluation: Optional[Dict] = None


@dataclass
class Experiment:
    id: str
    project_id: str
    config: Dict[str, Any]
    status: str = "running"
    started: str = field(default_factory=lambda: datetime.now().isoformat())
    completed: Optional[str] = None
    results: Optional[Dict] = None
    checkpoints: List[Dict] = field(default_factory=list)
    git_commit: str = ""


@dataclass
class DailyLog:
    date: str
    project_id: str
    phase: int
    notes: str
    blockers: str = ""
    time_spent_hours: float = 0


# ==================== STORAGE ====================

class LocalStorage:
    """JSON file-based storage."""
    
    @staticmethod
    def save(obj: Any, path: Path) -> None:
        """Save object to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load(path: Path, cls=None) -> Any:
        """Load object from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return cls(**data) if cls else data
    
    @staticmethod
    def list_all(directory: Path, cls=None) -> List[Any]:
        """List all objects in directory."""
        items = []
        for f in directory.glob('*.json'):
            try:
                items.append(LocalStorage.load(f, cls))
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
        return items


class GoogleSheetsSync:
    """Sync with Google Sheets (optional)."""
    
    def __init__(self):
        self.enabled = HAS_GSPREAD and Config.SHEET_ID
        self.sheet = None
        
        if self.enabled:
            try:
                # Try service account first, then API key
                self.gc = gspread.service_account() if os.path.exists('service_account.json') else None
                if self.gc:
                    self.sheet = self.gc.open_by_key(Config.SHEET_ID)
            except Exception as e:
                print(f"Google Sheets disabled: {e}")
                self.enabled = False
    
    def append_row(self, worksheet_name: str, row: List[Any]) -> None:
        """Append row to worksheet."""
        if not self.enabled:
            return
        try:
            ws = self.sheet.worksheet(worksheet_name)
            ws.append_row(row)
        except Exception as e:
            print(f"Sheets sync failed: {e}")
    
    def update_row(self, worksheet_name: str, id_col: int, id_value: str, row: List[Any]) -> None:
        """Update row by ID."""
        if not self.enabled:
            return
        try:
            ws = self.sheet.worksheet(worksheet_name)
            cell = ws.find(id_value, in_column=id_col)
            if cell:
                ws.update(f'A{cell.row}', [row])
        except Exception as e:
            print(f"Sheets update failed: {e}")


# ==================== EMAIL ====================

class EmailNotifier:
    """Send emails via Gmail SMTP."""
    
    def __init__(self):
        self.enabled = bool(Config.GMAIL_USER and Config.GMAIL_APP_PASSWORD)
    
    def send(self, subject: str, body: str, html: bool = False) -> bool:
        """Send email to researcher."""
        if not self.enabled:
            print("Email disabled (no credentials)")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = Config.GMAIL_USER
            msg['To'] = Config.GMAIL_USER
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(Config.GMAIL_USER, Config.GMAIL_APP_PASSWORD)
                server.send_message(msg)
            
            print(f"✉️ Email sent: {subject}")
            return True
        except Exception as e:
            print(f"Email failed: {e}")
            return False


# ==================== CORE TRACKER ====================

class VertexSimple:
    """Simple VERTEX-RESEARCH tracker."""
    
    def __init__(self):
        Config.ensure_dirs()
        self.storage = LocalStorage()
        self.sheets = GoogleSheetsSync()
        self.email = EmailNotifier()
        self.current_project_id: Optional[str] = None
        
        # Load current project if exists
        state_file = Config.VERTEX_DIR / 'current_state.json'
        if state_file.exists():
            state = self.storage.load(state_file)
            self.current_project_id = state.get('current_project_id')
    
    def _save_state(self) -> None:
        """Save current state."""
        self.storage.save(
            {'current_project_id': self.current_project_id},
            Config.VERTEX_DIR / 'current_state.json'
        )
    
    def _get_git_commit(self) -> str:
        """Get current git commit."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, cwd=Config.BASE_DIR
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    # ==================== PROJECTS ====================
    
    def project_new(self, name: str) -> Project:
        """Create new research project."""
        project_id = f"PROJ-{datetime.now().strftime('%Y%m%d')}-{name.lower().replace(' ', '-')[:20]}"
        
        project = Project(id=project_id, name=name)
        
        # Save locally
        self.storage.save(project, Config.PROJECTS_DIR / f"{project_id}.json")
        
        # Sync to sheets
        self.sheets.append_row('Projects', [
            project.id, project.name, project.phase, project.status, 
            project.created, project.notes
        ])
        
        # Set as current
        self.current_project_id = project_id
        self._save_state()
        
        print(f"✅ Project created: {project_id}")
        print(f"   Name: {name}")
        print(f"   Phase: {project.phase_name}")
        
        return project
    
    def project_list(self) -> List[Project]:
        """List all projects."""
        projects = self.storage.list_all(Config.PROJECTS_DIR, Project)
        
        print("\n📋 PROJECTS")
        print("=" * 60)
        for p in projects:
            status_emoji = "🟢" if p.status == "active" else "⏸️"
            current = " ← CURRENT" if p.id == self.current_project_id else ""
            print(f"{status_emoji} {p.name}{current}")
            print(f"   ID: {p.id}")
            print(f"   {p.phase_name}")
            print()
        
        return projects
    
    def project_advance(self, project_id: Optional[str] = None) -> Project:
        """Advance project to next phase."""
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project selected")
        
        path = Config.PROJECTS_DIR / f"{project_id}.json"
        project = self.storage.load(path, Project)
        
        if project.phase < len(Project.PHASES) - 1:
            project.phase += 1
            self.storage.save(project, path)
            
            # Sync to sheets
            self.sheets.update_row('Projects', 1, project.id, [
                project.id, project.name, project.phase, project.status,
                project.created, project.notes
            ])
            
            print(f"✅ Advanced to {project.phase_name}")
        else:
            print("⚠️ Already at final phase")
        
        return project
    
    # ==================== HYPOTHESES ====================
    
    def hypothesis_register(
        self,
        statement: str,
        conditions: str,
        falsification: str,
        thresholds: Optional[Dict[str, float]] = None,
        project_id: Optional[str] = None
    ) -> Hypothesis:
        """Register a falsifiable hypothesis."""
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project selected")
        
        hypothesis_id = f"H{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        hypothesis = Hypothesis(
            id=hypothesis_id,
            project_id=project_id,
            statement=statement,
            conditions=[c.strip() for c in conditions.split(',')],
            falsification_criteria=[f.strip() for f in falsification.split(',')],
            thresholds=thresholds or {}
        )
        
        # Save locally
        self.storage.save(hypothesis, Config.HYPOTHESES_DIR / f"{hypothesis_id}.json")
        
        # Sync to sheets
        self.sheets.append_row('Hypotheses', [
            hypothesis.id, hypothesis.project_id, hypothesis.statement,
            conditions, falsification, hypothesis.status, hypothesis.registered
        ])
        
        print(f"✅ Hypothesis registered: {hypothesis_id}")
        print(f"   Statement: {statement}")
        print(f"   Conditions: {hypothesis.conditions}")
        print(f"   Falsification: {hypothesis.falsification_criteria}")
        
        return hypothesis
    
    def hypothesis_evaluate(
        self,
        hypothesis_id: str,
        results: Dict[str, float]
    ) -> Dict[str, Any]:
        """Evaluate hypothesis against results."""
        path = Config.HYPOTHESES_DIR / f"{hypothesis_id}.json"
        hypothesis = self.storage.load(path, Hypothesis)
        
        evaluation = {
            'status': 'corroborated',
            'checks': [],
            'evaluated_at': datetime.now().isoformat()
        }
        
        # Check thresholds
        for metric, threshold in hypothesis.thresholds.items():
            if metric in results:
                actual = results[metric]
                # Assume 'min_' prefix means >= threshold, otherwise <= 
                if metric.startswith('min_'):
                    passed = actual >= threshold
                else:
                    passed = actual <= threshold
                
                if not passed:
                    evaluation['status'] = 'falsified'
                
                evaluation['checks'].append({
                    'metric': metric,
                    'threshold': threshold,
                    'actual': actual,
                    'passed': passed
                })
        
        # Update hypothesis
        hypothesis.status = evaluation['status']
        hypothesis.evaluation = evaluation
        self.storage.save(hypothesis, path)
        
        status_emoji = '✅' if evaluation['status'] == 'corroborated' else '❌'
        print(f"{status_emoji} Hypothesis {hypothesis_id}: {evaluation['status']}")
        for check in evaluation['checks']:
            check_emoji = '✓' if check['passed'] else '✗'
            print(f"   {check_emoji} {check['metric']}: {check['actual']} (threshold: {check['threshold']})")
        
        return evaluation
    
    # ==================== EXPERIMENTS ====================
    
    def experiment_start(
        self,
        config: Dict[str, Any],
        project_id: Optional[str] = None
    ) -> Experiment:
        """Start new experiment."""
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project selected")
        
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
        experiment_id = f"EXP-{datetime.now().strftime('%Y%m%d-%H%M')}-{config_hash}"
        
        experiment = Experiment(
            id=experiment_id,
            project_id=project_id,
            config=config,
            git_commit=self._get_git_commit()
        )
        
        # Save locally
        self.storage.save(experiment, Config.EXPERIMENTS_DIR / f"{experiment_id}.json")
        
        # Sync to sheets
        self.sheets.append_row('Experiments', [
            experiment.id, experiment.project_id, json.dumps(config),
            experiment.status, experiment.started, '', ''
        ])
        
        print(f"🚀 Experiment started: {experiment_id}")
        print(f"   Config: {json.dumps(config, indent=2)}")
        print(f"   Git: {experiment.git_commit}")
        
        return experiment
    
    def experiment_checkpoint(
        self,
        experiment_id: str,
        progress: float,
        metrics: Dict[str, float],
        notes: str = ""
    ) -> None:
        """Log experiment checkpoint."""
        path = Config.EXPERIMENTS_DIR / f"{experiment_id}.json"
        experiment = self.storage.load(path, Experiment)
        
        checkpoint = {
            'progress': progress,
            'metrics': metrics,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        experiment.checkpoints.append(checkpoint)
        
        self.storage.save(experiment, path)
        
        print(f"📊 Checkpoint: {progress:.0f}% | {metrics}")
    
    def experiment_complete(
        self,
        experiment_id: str,
        results: Dict[str, Any]
    ) -> Experiment:
        """Complete experiment with results."""
        path = Config.EXPERIMENTS_DIR / f"{experiment_id}.json"
        experiment = self.storage.load(path, Experiment)
        
        experiment.status = 'complete'
        experiment.completed = datetime.now().isoformat()
        experiment.results = results
        
        self.storage.save(experiment, path)
        
        # Sync to sheets
        self.sheets.update_row('Experiments', 1, experiment.id, [
            experiment.id, experiment.project_id, json.dumps(experiment.config),
            experiment.status, experiment.started, experiment.completed, json.dumps(results)
        ])
        
        print(f"✅ Experiment complete: {experiment_id}")
        print(f"   Results: {json.dumps(results, indent=2)}")
        
        return experiment
    
    # ==================== DAILY LOG ====================
    
    def daily_log(
        self,
        notes: str,
        blockers: str = "",
        hours: float = 0,
        project_id: Optional[str] = None
    ) -> DailyLog:
        """Create daily log entry."""
        project_id = project_id or self.current_project_id
        
        # Get current project phase
        phase = 0
        if project_id:
            try:
                project = self.storage.load(Config.PROJECTS_DIR / f"{project_id}.json", Project)
                phase = project.phase
            except:
                pass
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        log = DailyLog(
            date=today,
            project_id=project_id or "none",
            phase=phase,
            notes=notes,
            blockers=blockers,
            time_spent_hours=hours
        )
        
        # Save as markdown for easy reading
        md_content = f"""# Daily Log - {today}

**Project:** {project_id or 'None'}
**Phase:** {phase}
**Time Spent:** {hours}h

## Notes
{notes}

## Blockers
{blockers or 'None'}

---
*Generated by VERTEX-RESEARCH Protocol*
"""
        
        md_path = Config.DAILY_DIR / f"{today}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # Also save JSON
        self.storage.save(log, Config.DAILY_DIR / f"{today}.json")
        
        # Sync to sheets
        self.sheets.append_row('Daily Log', [
            log.date, log.project_id, log.phase, log.notes, log.blockers
        ])
        
        print(f"📝 Daily log saved: {today}")
        
        return log
    
    # ==================== REPORTS ====================
    
    def report_weekly(self) -> str:
        """Generate and send weekly report."""
        # Get all data from past week
        week_ago = datetime.now() - timedelta(days=7)
        
        projects = self.storage.list_all(Config.PROJECTS_DIR, Project)
        active_projects = [p for p in projects if p.status == 'active']
        
        experiments = self.storage.list_all(Config.EXPERIMENTS_DIR, Experiment)
        recent_experiments = [
            e for e in experiments 
            if datetime.fromisoformat(e.started) > week_ago
        ]
        completed_experiments = [e for e in recent_experiments if e.status == 'complete']
        
        hypotheses = self.storage.list_all(Config.HYPOTHESES_DIR, Hypothesis)
        recent_hypotheses = [
            h for h in hypotheses
            if datetime.fromisoformat(h.registered) > week_ago
        ]
        
        # Generate report
        report = f"""
📈 VERTEX WEEKLY REPORT
{'=' * 50}
Week ending: {datetime.now().strftime('%Y-%m-%d')}

📊 METRICS
{'─' * 30}
• Active projects: {len(active_projects)}
• Experiments started: {len(recent_experiments)}
• Experiments completed: {len(completed_experiments)}
• Hypotheses registered: {len(recent_hypotheses)}

🔬 PROJECTS STATUS
{'─' * 30}
"""
        
        for p in active_projects:
            report += f"""
{p.name}
├── Phase: {p.phase_name}
└── Status: {p.status}
"""
        
        if completed_experiments:
            report += f"""
✅ COMPLETED EXPERIMENTS
{'─' * 30}
"""
            for e in completed_experiments:
                report += f"• {e.id}: {json.dumps(e.results)}\n"
        
        report += f"""
{'─' * 50}
Report generated by VERTEX-RESEARCH Protocol
"""
        
        print(report)
        
        # Send email
        self.email.send(
            f"📈 VERTEX Weekly Report - {datetime.now().strftime('%Y-%m-%d')}",
            report
        )
        
        return report
    
    def report_daily_standup(self) -> str:
        """Generate daily standup."""
        projects = self.storage.list_all(Config.PROJECTS_DIR, Project)
        active_projects = [p for p in projects if p.status == 'active']
        
        # Check for recent blockers
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        recent_log = None
        for date in [yesterday, today]:
            log_path = Config.DAILY_DIR / f"{date}.json"
            if log_path.exists():
                recent_log = self.storage.load(log_path, DailyLog)
                break
        
        standup = f"""
🔬 VERTEX DAILY STANDUP
{'=' * 50}
{datetime.now().strftime('%A, %d %B %Y')}

Buongiorno {Config.RESEARCHER_NAME}!

📊 STATO PROGETTI
{'─' * 30}
"""
        
        for p in active_projects:
            standup += f"• {p.name}: {p.phase_name}\n"
        
        if recent_log and recent_log.blockers:
            standup += f"""
⚠️ BLOCKERS ATTIVI
{'─' * 30}
{recent_log.blockers}
"""
        else:
            standup += f"""
✅ Nessun blocker attivo
"""
        
        standup += f"""
{'─' * 50}
Buon lavoro!
"""
        
        print(standup)
        
        # Send email
        self.email.send(
            f"🔬 VERTEX Daily - {datetime.now().strftime('%Y-%m-%d')}",
            standup
        )
        
        return standup
    
    # ==================== GATE REVIEW ====================
    
    def gate_review(self, phase: int, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform gate review for phase transition."""
        project_id = project_id or self.current_project_id
        if not project_id:
            raise ValueError("No project selected")
        
        # Phase checklists
        checklists = {
            0: ["Problem statement defined", "Significance established", "Scope bounded", "Success criteria specified"],
            1: ["Literature review complete", "Gap analysis done", "References collected (>10 papers)"],
            2: ["Mathematical framework defined", "Assumptions documented", "Notation consistent"],
            3: ["Hypotheses falsifiable", "Thresholds quantified", "Pre-registered before experiments"],
            4: ["Experimental matrix complete", "Controls defined", "Statistical plan ready"],
            5: ["Code committed", "Tests passing", "Dependencies pinned", "Seeds fixed"],
            6: ["All seeds complete (5+)", "Logs archived", "No anomalies"],
            7: ["Falsification criteria checked", "Statistical tests done", "Limitations documented"],
            8: ["Results interpreted", "Visualizations created", "Insights extracted"],
            9: ["Paper draft complete", "Figures high quality", "References verified", "Proofread"],
        }
        
        checklist = checklists.get(phase, [])
        
        print(f"\n🔍 GATE REVIEW - Phase {phase}")
        print("=" * 50)
        print(f"Project: {project_id}")
        print()
        print("Checklist:")
        
        results = {'phase': phase, 'items': [], 'passed': True}
        
        for item in checklist:
            response = input(f"  [ ] {item}? (y/n/s): ").strip().lower()
            passed = response == 'y'
            skipped = response == 's'
            
            results['items'].append({
                'item': item,
                'passed': passed,
                'skipped': skipped
            })
            
            if not passed and not skipped:
                results['passed'] = False
        
        print()
        if results['passed']:
            print("✅ GATE REVIEW PASSED")
            print("   You can advance to the next phase.")
            
            advance = input("   Advance now? (y/n): ").strip().lower()
            if advance == 'y':
                self.project_advance(project_id)
        else:
            print("⚠️ GATE REVIEW NEEDS ATTENTION")
            failed_items = [i['item'] for i in results['items'] if not i['passed'] and not i['skipped']]
            print(f"   Missing: {', '.join(failed_items)}")
        
        return results


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='VERTEX-RESEARCH Simple Tracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s project new "My Research"
  %(prog)s hypothesis register --statement "X improves Y by >25%%"
  %(prog)s experiment start --config '{"model": "test"}'
  %(prog)s daily --notes "Completed phase 2"
  %(prog)s report weekly
  %(prog)s gate review --phase 3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Project commands
    project_parser = subparsers.add_parser('project', help='Project management')
    project_sub = project_parser.add_subparsers(dest='action')
    
    project_new = project_sub.add_parser('new', help='Create new project')
    project_new.add_argument('name', help='Project name')
    
    project_sub.add_parser('list', help='List all projects')
    
    project_advance = project_sub.add_parser('advance', help='Advance to next phase')
    project_advance.add_argument('--id', help='Project ID (default: current)')
    
    # Hypothesis commands
    hypo_parser = subparsers.add_parser('hypothesis', help='Hypothesis management')
    hypo_sub = hypo_parser.add_subparsers(dest='action')
    
    hypo_reg = hypo_sub.add_parser('register', help='Register hypothesis')
    hypo_reg.add_argument('--statement', required=True, help='Hypothesis statement')
    hypo_reg.add_argument('--conditions', required=True, help='Conditions (comma-separated)')
    hypo_reg.add_argument('--falsification', required=True, help='Falsification criteria (comma-separated)')
    hypo_reg.add_argument('--thresholds', help='JSON thresholds')
    
    hypo_eval = hypo_sub.add_parser('evaluate', help='Evaluate hypothesis')
    hypo_eval.add_argument('--id', required=True, help='Hypothesis ID')
    hypo_eval.add_argument('--results', required=True, help='JSON results')
    
    # Experiment commands
    exp_parser = subparsers.add_parser('experiment', help='Experiment tracking')
    exp_sub = exp_parser.add_subparsers(dest='action')
    
    exp_start = exp_sub.add_parser('start', help='Start experiment')
    exp_start.add_argument('--config', required=True, help='JSON config')
    
    exp_check = exp_sub.add_parser('checkpoint', help='Log checkpoint')
    exp_check.add_argument('--id', required=True, help='Experiment ID')
    exp_check.add_argument('--progress', type=float, required=True, help='Progress (0-100)')
    exp_check.add_argument('--metrics', required=True, help='JSON metrics')
    
    exp_complete = exp_sub.add_parser('complete', help='Complete experiment')
    exp_complete.add_argument('--id', required=True, help='Experiment ID')
    exp_complete.add_argument('--results', required=True, help='JSON results')
    
    # Daily log
    daily_parser = subparsers.add_parser('daily', help='Daily log')
    daily_parser.add_argument('--notes', required=True, help='Daily notes')
    daily_parser.add_argument('--blockers', default='', help='Blockers')
    daily_parser.add_argument('--hours', type=float, default=0, help='Hours spent')
    
    # Reports
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_sub = report_parser.add_subparsers(dest='action')
    report_sub.add_parser('weekly', help='Weekly report')
    report_sub.add_parser('standup', help='Daily standup')
    
    # Gate review
    gate_parser = subparsers.add_parser('gate', help='Gate reviews')
    gate_sub = gate_parser.add_subparsers(dest='action')
    gate_review = gate_sub.add_parser('review', help='Perform gate review')
    gate_review.add_argument('--phase', type=int, required=True, help='Phase number')
    
    args = parser.parse_args()
    
    tracker = VertexSimple()
    
    # Route commands
    if args.command == 'project':
        if args.action == 'new':
            tracker.project_new(args.name)
        elif args.action == 'list':
            tracker.project_list()
        elif args.action == 'advance':
            tracker.project_advance(args.id)
    
    elif args.command == 'hypothesis':
        if args.action == 'register':
            thresholds = json.loads(args.thresholds) if args.thresholds else None
            tracker.hypothesis_register(
                args.statement, args.conditions, args.falsification, thresholds
            )
        elif args.action == 'evaluate':
            tracker.hypothesis_evaluate(args.id, json.loads(args.results))
    
    elif args.command == 'experiment':
        if args.action == 'start':
            tracker.experiment_start(json.loads(args.config))
        elif args.action == 'checkpoint':
            tracker.experiment_checkpoint(
                args.id, args.progress, json.loads(args.metrics)
            )
        elif args.action == 'complete':
            tracker.experiment_complete(args.id, json.loads(args.results))
    
    elif args.command == 'daily':
        tracker.daily_log(args.notes, args.blockers, args.hours)
    
    elif args.command == 'report':
        if args.action == 'weekly':
            tracker.report_weekly()
        elif args.action == 'standup':
            tracker.report_daily_standup()
    
    elif args.command == 'gate':
        if args.action == 'review':
            tracker.gate_review(args.phase)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
