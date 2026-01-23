#!/usr/bin/env python3
"""
AV Simulation at Intersections - Interactive GUI Application
A modern, parameter-driven interface for running simulations and analyses.
"""

import os
import sys
import subprocess
import threading
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
MAIN_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(MAIN_DIR))

# Install customtkinter if not available
try:
    import customtkinter as ctk
except ImportError:
    print("Installing CustomTkinter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter", "-q"])
    import customtkinter as ctk

try:
    from CTkToolTip import CTkToolTip
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "CTkToolTip", "-q"])
    from CTkToolTip import CTkToolTip

from tkinter import messagebox


# ============================================================================
# CONFIGURATION
# ============================================================================

# Mapping from readable names to internal values
POSITION_MAP = {"South": 1, "West": 2, "North": 3, "East": 4}
TURN_MAP = {"Left": 1, "Straight": 2, "Right": 3}
LANE_MAP = {"Inner": 1, "Outer": 2}
TURN_MAP_ROUNDABOUT = {"Right (1st exit)": 3, "Straight (2nd exit)": 2, "Left (3rd exit)": 1, "U-turn (4th exit)": 4}

PLANNER_SCENARIOS = {
    "T-Intersection (Simple)": {
        "script": "planner/motion_primitive_search.py",
        "description": "Basic T-intersection with turn left/right option",
        "params": {
            "turn_left": {"type": "bool", "default": True, "label": "Turn Left"},
        }
    },
    "Intersection (4-way)": {
        "script": "planner/motion_primitive_search_full_intersection.py",
        "description": "Full 4-way intersection with multiple start positions",
        "params": {
            "start_pos": {"type": "choice", "options": ["South", "West", "North", "East"], "default": "South", 
                          "label": "Start Position", "map": POSITION_MAP},
            "turn_indicator": {"type": "choice", "options": ["Left", "Straight", "Right"], "default": "Left",
                               "label": "Turn Direction", "map": TURN_MAP},
        }
    },
    "Multi-lane Intersection": {
        "script": "planner/motion_primitive_search_full_intersection_multi_lane.py",
        "description": "Multi-lane intersection with lane selection",
        "params": {
            "start_pos": {"type": "choice", "options": ["South", "West", "North", "East"], "default": "South", 
                          "label": "Start Position", "map": POSITION_MAP},
            "turn_indicator": {"type": "choice", "options": ["Left", "Straight", "Right"], "default": "Left",
                               "label": "Turn Direction", "map": TURN_MAP},
            "start_lane": {"type": "choice", "options": ["Inner", "Outer"], "default": "Inner", 
                           "label": "Start Lane", "map": LANE_MAP},
            "goal_lane": {"type": "choice", "options": ["Inner", "Outer"], "default": "Inner", 
                          "label": "Goal Lane", "map": LANE_MAP},
        }
    },
    "Roundabout": {
        "script": "planner/motion_primitive_search_roundabout.py",
        "description": "Roundabout navigation scenario",
        "params": {
            "start_pos": {"type": "choice", "options": ["South", "West", "North", "East"], "default": "South", 
                          "label": "Entry Position", "map": POSITION_MAP},
            "turn_indicator": {"type": "choice", "options": ["Right (1st exit)", "Straight (2nd exit)", "Left (3rd exit)", "U-turn (4th exit)"], 
                               "default": "U-turn (4th exit)", "label": "Exit Direction", "map": TURN_MAP_ROUNDABOUT},
            "size": {"type": "choice", "options": ["Normal", "Big"], "default": "Big", "label": "Roundabout Size"},
        }
    },
}

PLANNER_PARAMS = {
    "Heuristic Weights": {
        "wh_dist": {"type": "slider", "min": 0, "max": 5, "default": 1, "label": "Distance Weight (Wh,d)"},
        "wh_theta": {"type": "slider", "min": 0, "max": 5, "default": 2.7, "label": "Heading Weight (Wh,θ)"},
        "wh_steering": {"type": "slider", "min": 0, "max": 20, "default": 0, "label": "Steering Weight (Wh,φ)"},
    }
}

CONTROLLER_SCENARIOS = {
    "Intersection": {
        "script": "scenarios/mpc_intersection.py",
        "description": "MPC controller for intersection navigation",
        "params": {
            "start_pos": {"type": "choice", "options": ["South", "West", "North", "East"], "default": "South", 
                          "label": "Start Position", "map": POSITION_MAP},
            "turn_indicator": {"type": "choice", "options": ["Left", "Straight", "Right"], "default": "Left",
                               "label": "Turn Direction", "map": TURN_MAP},
        }
    },
    "Roundabout": {
        "script": "scenarios/mpc_roundabout.py",
        "description": "MPC controller for roundabout navigation",
        "params": {
            "start_pos": {"type": "choice", "options": ["South", "West", "North", "East"], "default": "South", 
                          "label": "Entry Position", "map": POSITION_MAP},
            "turn_indicator": {"type": "choice", "options": ["Right (1st exit)", "Straight (2nd exit)", "Left (3rd exit)", "U-turn (4th exit)"], 
                               "default": "U-turn (4th exit)", "label": "Exit Direction", "map": TURN_MAP_ROUNDABOUT},
        }
    },
    "Multi-lane Intersection": {
        "script": "scenarios/mpc_intersection_multi_lane.py",
        "description": "MPC for multi-lane intersection",
        "params": {
            "start_pos": {"type": "choice", "options": ["South", "West", "North", "East"], "default": "South", 
                          "label": "Start Position", "map": POSITION_MAP},
            "turn_indicator": {"type": "choice", "options": ["Left", "Straight", "Right"], "default": "Left",
                               "label": "Turn Direction", "map": TURN_MAP},
        }
    },
    "Basic Test": {
        "script": "scenarios/mpc_basic.py",
        "description": "Simple MPC test scenario",
        "params": {}
    },
}

MPC_PARAMS = {
    "Cost Weights": {
        "w_perp": {"type": "slider", "min": 0.1, "max": 50, "default": 10, "label": "Perpendicular Error (w⊥)"},
        "w_para": {"type": "slider", "min": 0.1, "max": 50, "default": 1, "label": "Parallel Error (w∥)"},
    },
    "Control Limits": {
        "MAX_ACCEL": {"type": "slider", "min": 0.5, "max": 5, "default": 2, "label": "Max Acceleration (m/s²)"},
        "MAX_DECEL": {"type": "slider", "min": -10, "max": -1, "default": -5, "label": "Max Deceleration (m/s²)"},
        "MAX_DSTEER": {"type": "slider", "min": 10, "max": 60, "default": 30, "label": "Max Steering Rate (°/s)"},
    },
    "Horizon": {
        "T": {"type": "slider", "min": 5, "max": 25, "default": 13, "step": 1, "label": "Prediction Horizon"},
    }
}


# ============================================================================
# COLORS - Simple color scheme
# ============================================================================

# Accent colors (same in both themes)
COLOR_HIGHLIGHT = "#e94560"
COLOR_SUCCESS = "#00d9a5"
COLOR_WARNING = "#ffc107"
COLOR_PLANNER = "#3498db"
COLOR_CONTROLLER = "#e74c3c"
COLOR_TOOLS = "#9b59b6"


# ============================================================================
# OUTPUT WINDOW
# ============================================================================

class OutputWindow(ctk.CTkToplevel):
    """Window to display script output."""
    
    def __init__(self, parent, title="Output"):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x600")
        
        # Header
        header = ctk.CTkFrame(self, height=50)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text=f"📺 {title}", font=("Helvetica", 16, "bold")).pack(side="left", padx=20, pady=10)
        
        self.status_label = ctk.CTkLabel(header, text="● Running", text_color=COLOR_WARNING, font=("Helvetica", 12))
        self.status_label.pack(side="right", padx=20)
        
        # Output
        self.output_text = ctk.CTkTextbox(self, font=("Courier", 12))
        self.output_text.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        ctk.CTkButton(btn_frame, text="⬛ Stop", command=self.stop_process,
                      fg_color=COLOR_HIGHLIGHT, width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Close", command=self.destroy, width=100).pack(side="right", padx=5)
        
        self.process = None
    
    def append_text(self, text):
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.update()
    
    def stop_process(self):
        if self.process:
            self.process.terminate()
            self.append_text("\n⬛ Stopped\n")
            self.status_label.configure(text="● Stopped", text_color=COLOR_HIGHLIGHT)
    
    def run_script(self, script_path, params: Dict[str, Any] = None):
        def _run():
            env = os.environ.copy()
            env['PYTHONPATH'] = str(MAIN_DIR)
            
            if params:
                for key, value in params.items():
                    env[f'AV_PARAM_{key.upper()}'] = str(value)
            
            self.append_text(f"▶ Running: {script_path}\n")
            if params:
                self.append_text(f"📋 Parameters: {json.dumps(params, indent=2)}\n")
            self.append_text("=" * 60 + "\n\n")
            
            try:
                self.process = subprocess.Popen(
                    [sys.executable, str(MAIN_DIR / script_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, cwd=str(MAIN_DIR), text=True, bufsize=1
                )
                
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        self.after(0, lambda l=line: self.append_text(l))
                
                self.process.wait()
                code = self.process.returncode
                
                if code == 0:
                    self.after(0, lambda: self.status_label.configure(text="● Done", text_color=COLOR_SUCCESS))
                    self.after(0, lambda: self.append_text(f"\n{'=' * 60}\n✓ Success\n"))
                else:
                    self.after(0, lambda: self.status_label.configure(text="● Error", text_color=COLOR_HIGHLIGHT))
                    self.after(0, lambda: self.append_text(f"\n{'=' * 60}\n✗ Exit code: {code}\n"))
                    
            except Exception as e:
                self.after(0, lambda: self.append_text(f"\n❌ Error: {e}\n"))
        
        threading.Thread(target=_run, daemon=True).start()


# ============================================================================
# PARAMETER PANEL
# ============================================================================

class ParameterPanel(ctk.CTkFrame):
    """Panel for editing parameters."""
    
    def __init__(self, parent, title: str, params: Dict, color: str):
        super().__init__(parent, corner_radius=15)
        
        self.params = params
        self.widgets = {}
        
        # Header
        header = ctk.CTkFrame(self, fg_color=color, corner_radius=10, height=40)
        header.pack(fill="x", padx=10, pady=(10, 5))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text=f"⚙️  {title}", font=("Helvetica", 14, "bold"),
                     text_color="white").pack(side="left", padx=15, pady=8)
        
        for name, config in params.items():
            self._create_widget(name, config)
    
    def _create_widget(self, name: str, config: Dict):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=5)
        
        # Label - let CTk handle the text color automatically
        label = ctk.CTkLabel(frame, text=config["label"], font=("Helvetica", 13), width=200, anchor="w")
        label.pack(side="left")
        
        if config.get("help"):
            CTkToolTip(label, message=config["help"])
        
        if config["type"] == "slider":
            var = ctk.DoubleVar(value=config["default"])
            
            slider = ctk.CTkSlider(frame, from_=config["min"], to=config["max"], variable=var, width=180)
            slider.pack(side="left", padx=10)
            
            value_label = ctk.CTkLabel(frame, text=f"{config['default']:.1f}", width=60, 
                                        font=("Helvetica", 12, "bold"))
            value_label.pack(side="left")
            
            def update(v=var, l=value_label, s=config.get("step", 0.1)):
                if s >= 1:
                    l.configure(text=f"{int(v.get())}")
                else:
                    l.configure(text=f"{v.get():.1f}")
            var.trace_add("write", lambda *_: update())
            
            self.widgets[name] = var
            
        elif config["type"] == "choice":
            var = ctk.StringVar(value=str(config["default"]))
            ctk.CTkOptionMenu(frame, values=[str(o) for o in config["options"]], 
                              variable=var, width=150).pack(side="left", padx=10)
            self.widgets[name] = var
            
        elif config["type"] == "bool":
            var = ctk.BooleanVar(value=config["default"])
            ctk.CTkSwitch(frame, text="", variable=var).pack(side="left", padx=10)
            self.widgets[name] = var
    
    def get_values(self) -> Dict[str, Any]:
        result = {}
        for name, widget in self.widgets.items():
            val = widget.get()
            config = self.params[name]
            if config["type"] == "slider":
                result[name] = int(val) if config.get("step", 0.1) >= 1 else float(val)
            elif config["type"] == "choice":
                # Check if there's a mapping to convert readable names to values
                if "map" in config and val in config["map"]:
                    result[name] = config["map"][val]
                elif val.lower() in ["normal", "big"]:
                    result[name] = val.lower()  # Handle size option
                else:
                    try:
                        result[name] = int(val)
                    except ValueError:
                        result[name] = val
            else:
                result[name] = val
        return result


# ============================================================================
# SCENARIO SELECTOR
# ============================================================================

class ScenarioSelector(ctk.CTkFrame):
    """Dropdown to select scenario with dynamic parameters."""
    
    def __init__(self, parent, scenarios: Dict, color: str, on_run):
        super().__init__(parent, corner_radius=15)
        
        self.scenarios = scenarios
        self.color = color
        self.on_run = on_run
        self.param_panel = None
        
        # Selection
        select_frame = ctk.CTkFrame(self, fg_color="transparent")
        select_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(select_frame, text="📍 Scenario", font=("Helvetica", 15, "bold"),
                     width=120, anchor="w").pack(side="left")
        
        self.scenario_var = ctk.StringVar(value=list(scenarios.keys())[0])
        self.dropdown = ctk.CTkOptionMenu(
            select_frame, values=list(scenarios.keys()), variable=self.scenario_var,
            command=self._on_change, width=280, font=("Helvetica", 14),
            fg_color=color, button_color=color
        )
        self.dropdown.pack(side="left", padx=15)
        
        # Description
        self.desc_label = ctk.CTkLabel(self, text="", font=("Helvetica", 12))
        self.desc_label.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Params container
        self.param_container = ctk.CTkFrame(self, fg_color="transparent")
        self.param_container.pack(fill="x", padx=5)
        
        # Run button
        self.run_btn = ctk.CTkButton(
            self, text="▶  RUN SCENARIO", font=("Helvetica", 16, "bold"),
            command=self._run, height=50, fg_color=color, corner_radius=10
        )
        self.run_btn.pack(fill="x", padx=20, pady=20)
        
        self._on_change(self.scenario_var.get())
    
    def _on_change(self, name: str):
        scenario = self.scenarios[name]
        self.desc_label.configure(text=scenario["description"])
        
        for w in self.param_container.winfo_children():
            w.destroy()
        
        if scenario.get("params"):
            self.param_panel = ParameterPanel(self.param_container, "Scenario Parameters",
                                               scenario["params"], self.color)
            self.param_panel.pack(fill="x")
        else:
            self.param_panel = None
    
    def _run(self):
        name = self.scenario_var.get()
        scenario = self.scenarios[name]
        params = self.param_panel.get_values() if self.param_panel else {}
        self.on_run(scenario["script"], name, params)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class AVSimulationApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.title("🚗 AV Simulation at Intersections")
        self.geometry("1300x900")
        self.minsize(1100, 750)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self._build_ui()
    
    def _build_ui(self):
        """Build or rebuild the entire UI with current theme colors."""
        # Clear existing widgets
        for widget in self.winfo_children():
            widget.destroy()
        
        # ===== HEADER =====
        header = ctk.CTkFrame(self, height=80, corner_radius=0)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=40, pady=15)
        
        ctk.CTkLabel(title_frame, text="🚗", font=("Helvetica", 40)).pack(side="left")
        
        text_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        text_frame.pack(side="left", padx=20)
        
        ctk.CTkLabel(text_frame, text="AV Simulation at Intersections",
                     font=("Helvetica", 26, "bold")).pack(anchor="w")
        ctk.CTkLabel(text_frame, text="Interactive Motion Planning & Control Framework",
                     font=("Helvetica", 13)).pack(anchor="w")
        
        # Theme toggle
        theme_frame = ctk.CTkFrame(header, fg_color="transparent")
        theme_frame.pack(side="right", padx=40)
        
        ctk.CTkLabel(theme_frame, text="🌙", font=("Helvetica", 18)).pack(side="left", padx=5)
        self.theme_switch = ctk.CTkSwitch(theme_frame, text="", command=self._toggle_theme, width=40)
        self.theme_switch.pack(side="left")
        if ctk.get_appearance_mode() == "Light":
            self.theme_switch.select()
        ctk.CTkLabel(theme_frame, text="☀️", font=("Helvetica", 18)).pack(side="left", padx=5)
        
        # ===== TABS =====
        self.tabview = ctk.CTkTabview(
            self, corner_radius=15,
            segmented_button_selected_color=COLOR_HIGHLIGHT
        )
        self.tabview.pack(fill="both", expand=True, padx=25, pady=25)
        
        self._create_planner_tab()
        self._create_controller_tab()
        self._create_tools_tab()
    
    def _create_planner_tab(self):
        tab = self.tabview.add("🎯 Planner")
        
        # Two columns
        cols = ctk.CTkFrame(tab, fg_color="transparent")
        cols.pack(fill="both", expand=True, pady=10)
        
        left = ctk.CTkFrame(cols, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        right = ctk.CTkFrame(cols, fg_color="transparent")
        right.pack(side="right", fill="both", expand=True, padx=(15, 0))
        
        # Left: Scenarios
        ctk.CTkLabel(left, text="🗺️  Select Scenario", font=("Helvetica", 18, "bold")).pack(anchor="w", pady=(0, 15))
        
        ScenarioSelector(left, PLANNER_SCENARIOS, COLOR_PLANNER,
                         lambda s, n, p: self._run(s, n, p)).pack(fill="x")
        
        # Right: Parameters
        ctk.CTkLabel(right, text="⚙️  Planner Parameters", font=("Helvetica", 18, "bold")).pack(anchor="w", pady=(0, 15))
        
        scroll = ctk.CTkScrollableFrame(right, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        
        for section, params in PLANNER_PARAMS.items():
            ParameterPanel(scroll, section, params, COLOR_PLANNER).pack(fill="x", pady=8)
        
        # Sensitivity
        sens = ctk.CTkFrame(scroll, corner_radius=15)
        sens.pack(fill="x", pady=15)
        
        ctk.CTkLabel(sens, text="📈 Sensitivity Analysis", font=("Helvetica", 15, "bold")).pack(anchor="w", padx=20, pady=(20, 10))
        
        btns = ctk.CTkFrame(sens, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(btns, text="Heuristic Study", width=150, height=40,
                      command=lambda: self._run("planner/Planner_Sensitivity_Heuristic.py", "Sensitivity", {}),
                      fg_color=COLOR_PLANNER).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="True Cost Study", width=150, height=40,
                      command=lambda: self._run("planner/Planner_Sensitivity_TrueCost.py", "True Cost", {}),
                      fg_color=COLOR_PLANNER).pack(side="left", padx=5)
    
    def _create_controller_tab(self):
        tab = self.tabview.add("🎮 Controller")
        
        cols = ctk.CTkFrame(tab, fg_color="transparent")
        cols.pack(fill="both", expand=True, pady=10)
        
        left = ctk.CTkFrame(cols, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=(0, 15))
        
        right = ctk.CTkFrame(cols, fg_color="transparent")
        right.pack(side="right", fill="both", expand=True, padx=(15, 0))
        
        # Left: Scenarios
        ctk.CTkLabel(left, text="🗺️  Select Scenario", font=("Helvetica", 18, "bold")).pack(anchor="w", pady=(0, 15))
        
        ScenarioSelector(left, CONTROLLER_SCENARIOS, COLOR_CONTROLLER,
                         lambda s, n, p: self._run(s, n, p)).pack(fill="x")
        
        # Right: MPC Params
        ctk.CTkLabel(right, text="⚙️  MPC Parameters", font=("Helvetica", 18, "bold")).pack(anchor="w", pady=(0, 15))
        
        scroll = ctk.CTkScrollableFrame(right, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        
        for section, params in MPC_PARAMS.items():
            ParameterPanel(scroll, section, params, COLOR_CONTROLLER).pack(fill="x", pady=8)
        
        # Sensitivity
        sens = ctk.CTkFrame(scroll, corner_radius=15)
        sens.pack(fill="x", pady=15)
        
        ctk.CTkLabel(sens, text="📊 Sensitivity Analysis", font=("Helvetica", 15, "bold")).pack(anchor="w", padx=20, pady=(20, 10))
        
        btns = ctk.CTkFrame(sens, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(btns, text="Parameter Study", width=150, height=40,
                      command=lambda: self._run("scenarios/mpc_sensitivity_analysis_comulative.py", "MPC Sensitivity", {}),
                      fg_color=COLOR_CONTROLLER).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="Interactive MPC", width=150, height=40,
                      command=lambda: self._run("scenarios/interactive_mpc.py", "Interactive", {}),
                      fg_color=COLOR_CONTROLLER).pack(side="left", padx=5)
    
    def _create_tools_tab(self):
        tab = self.tabview.add("🔧 Tools")
        
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, pady=10)
        
        # MP Generation
        mp = ctk.CTkFrame(scroll, corner_radius=15)
        mp.pack(fill="x", pady=10)
        
        ctk.CTkLabel(mp, text="⚙️  Motion Primitive Generation", font=("Helvetica", 17, "bold")).pack(anchor="w", padx=25, pady=(25, 10))
        ctk.CTkLabel(mp, text="Generate motion primitives for vehicle models").pack(anchor="w", padx=25)
        
        btns = ctk.CTkFrame(mp, fg_color="transparent")
        btns.pack(fill="x", padx=25, pady=25)
        
        ctk.CTkButton(btns, text="Bicycle Model", width=180, height=45,
                      command=lambda: self._run("create_motion_primitives_bicycle_model.py", "Bicycle MPs", {}),
                      fg_color=COLOR_TOOLS).pack(side="left", padx=5)
        ctk.CTkButton(btns, text="Prius Model", width=180, height=45,
                      command=lambda: self._run("create_motion_primitives_prius.py", "Prius MPs", {}),
                      fg_color=COLOR_TOOLS).pack(side="left", padx=5)
        
        # Dependencies
        dep = ctk.CTkFrame(scroll, corner_radius=15)
        dep.pack(fill="x", pady=10)
        
        ctk.CTkLabel(dep, text="📦  Dependencies", font=("Helvetica", 17, "bold")).pack(anchor="w", padx=25, pady=(25, 10))
        
        btns2 = ctk.CTkFrame(dep, fg_color="transparent")
        btns2.pack(fill="x", padx=25, pady=(0, 25))
        
        ctk.CTkButton(btns2, text="Check All", width=180, height=45,
                      command=self._check_deps, fg_color=COLOR_SUCCESS).pack(side="left", padx=5)
        ctk.CTkButton(btns2, text="Install Missing", width=180, height=45,
                      command=self._install_deps).pack(side="left", padx=5)
        
        # About
        about = ctk.CTkFrame(scroll, corner_radius=15)
        about.pack(fill="x", pady=10)
        
        ctk.CTkLabel(about, text="ℹ️  About", font=("Helvetica", 17, "bold")).pack(anchor="w", padx=25, pady=(25, 10))
        
        info = (
            "AV Simulation at Intersections\n\n"
            "A bi-level framework for modeling 2D vehicular movements and\n"
            "interactions at urban junctions.\n\n"
            "• Motion Primitive Search (A* based path planning)\n"
            "• Model Predictive Control (trajectory tracking)\n"
            "• Environments: Intersection, Roundabout, T-junction\n\n"
            "Requires Python 3.8 - 3.11"
        )
        ctk.CTkLabel(about, text=info, font=("Helvetica", 13),
                     justify="left").pack(anchor="w", padx=25, pady=(0, 25))
    
    def _run(self, script: str, name: str, params: Dict):
        OutputWindow(self, title=name).run_script(script, params)
    
    def _toggle_theme(self):
        """Toggle between dark and light theme and rebuild UI."""
        new_mode = "light" if ctk.get_appearance_mode() == "Dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        # Rebuild UI with new theme colors
        self._build_ui()
    
    def _check_deps(self):
        required = ['numpy', 'matplotlib', 'cvxpy', 'tqdm', 'networkx', 'ecos']
        missing = [p for p in required if not self._is_installed(p)]
        
        if missing:
            messagebox.showwarning("Missing", f"Missing: {', '.join(missing)}")
        else:
            messagebox.showinfo("OK", "✓ All dependencies installed!")
    
    def _install_deps(self):
        req = MAIN_DIR.parent / "requirements.txt"
        if req.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req)])
            messagebox.showinfo("Done", "Dependencies installed!")
    
    def _is_installed(self, pkg):
        try:
            __import__(pkg)
            return True
        except ImportError:
            return False


if __name__ == "__main__":
    AVSimulationApp().mainloop()
