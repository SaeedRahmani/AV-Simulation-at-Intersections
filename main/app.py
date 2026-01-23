#!/usr/bin/env python3
"""
AV Simulation at Intersections - Modern GUI Application
A sleek, modern interface for running simulations and analyses.
"""

import os
import sys
import subprocess
import threading
from pathlib import Path

# Add parent directory to path
MAIN_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(MAIN_DIR))

# Install customtkinter if not available
try:
    import customtkinter as ctk
except ImportError:
    print("Installing CustomTkinter (modern UI library)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter", "-q"])
    import customtkinter as ctk

from tkinter import messagebox
import tkinter as tk


class OutputWindow(ctk.CTkToplevel):
    """Window to display script output."""
    
    def __init__(self, parent, title="Output"):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x500")
        
        # Output text area
        self.output_text = ctk.CTkTextbox(self, font=("Courier", 12))
        self.output_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Close button
        self.close_btn = ctk.CTkButton(
            self, text="Close", command=self.destroy,
            fg_color="#e74c3c", hover_color="#c0392b"
        )
        self.close_btn.pack(pady=10)
        
        self.process = None
    
    def append_text(self, text):
        """Append text to the output area."""
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.update()
    
    def run_script(self, script_path):
        """Run a script and display output."""
        def _run():
            env = os.environ.copy()
            env['PYTHONPATH'] = str(MAIN_DIR)
            
            self.append_text(f"▶ Running: {script_path}\n")
            self.append_text("=" * 50 + "\n\n")
            
            try:
                self.process = subprocess.Popen(
                    [sys.executable, str(MAIN_DIR / script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(MAIN_DIR),
                    text=True,
                    bufsize=1
                )
                
                for line in iter(self.process.stdout.readline, ''):
                    if line:
                        self.after(0, lambda l=line: self.append_text(l))
                
                self.process.wait()
                self.after(0, lambda: self.append_text(f"\n{'=' * 50}\n✓ Completed with code: {self.process.returncode}\n"))
                
            except Exception as e:
                self.after(0, lambda: self.append_text(f"\n❌ Error: {e}\n"))
        
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()


class ScriptButton(ctk.CTkButton):
    """Custom button for running scripts."""
    
    def __init__(self, parent, text, script_path, description, app, **kwargs):
        self.script_path = script_path
        self.description = description
        self.app = app
        
        super().__init__(
            parent,
            text=text,
            command=self.run,
            height=45,
            font=("Helvetica", 14),
            anchor="w",
            **kwargs
        )
    
    def run(self):
        """Run the associated script."""
        output_win = OutputWindow(self.app, title=self.description)
        output_win.run_script(self.script_path)


class CategoryFrame(ctk.CTkFrame):
    """Frame for a category of scripts."""
    
    def __init__(self, parent, title, icon, color, scripts, app):
        super().__init__(parent, corner_radius=15)
        
        # Header
        header = ctk.CTkFrame(self, fg_color=color, corner_radius=10)
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        header_label = ctk.CTkLabel(
            header,
            text=f"{icon}  {title}",
            font=("Helvetica", 18, "bold"),
            text_color="white"
        )
        header_label.pack(pady=10, padx=15, anchor="w")
        
        # Scripts
        for name, script, desc in scripts:
            btn = ScriptButton(
                self,
                text=f"  {name}",
                script_path=script,
                description=desc,
                app=app,
                fg_color="transparent",
                hover_color=("gray85", "gray25"),
                text_color=("gray10", "gray90")
            )
            btn.pack(fill="x", padx=15, pady=3)


class AVSimulationApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title("🚗 AV Simulation at Intersections")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all UI widgets."""
        
        # Header
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="🚗 AV Simulation at Intersections",
            font=("Helvetica", 28, "bold")
        )
        title_label.pack(side="left")
        
        # Theme toggle
        self.theme_var = ctk.StringVar(value="dark")
        theme_switch = ctk.CTkSwitch(
            header_frame,
            text="Light Mode",
            variable=self.theme_var,
            onvalue="light",
            offvalue="dark",
            command=self.toggle_theme
        )
        theme_switch.pack(side="right", padx=10)
        
        subtitle = ctk.CTkLabel(
            self,
            text="Motion Planning & Control for Autonomous Vehicles",
            font=("Helvetica", 14),
            text_color="gray"
        )
        subtitle.pack(anchor="w", padx=25)
        
        # Main content with tabs
        self.tabview = ctk.CTkTabview(self, corner_radius=15)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create tabs
        self.create_planner_tab()
        self.create_controller_tab()
        self.create_simulation_tab()
        self.create_tools_tab()
    
    def create_planner_tab(self):
        """Create the Planner tab."""
        tab = self.tabview.add("🎯 Planner")
        
        # Description
        desc = ctk.CTkLabel(
            tab,
            text="Motion primitive search algorithms for path planning",
            font=("Helvetica", 13),
            text_color="gray"
        )
        desc.pack(anchor="w", padx=10, pady=(10, 15))
        
        # Scrollable frame for content
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)
        
        # Motion Primitive Search
        mp_scripts = [
            ("Intersection", "planner/motion_primitive_search.py", "Motion Primitive Search - Intersection"),
            ("Roundabout", "planner/motion_primitive_search_roundabout.py", "Motion Primitive Search - Roundabout"),
            ("Multi-lane Intersection", "planner/motion_primitive_search_full_intersection_multi_lane.py", "Multi-lane Intersection"),
            ("Full Intersection", "planner/motion_primitive_search_full_intersection.py", "Full Intersection Search"),
        ]
        
        mp_frame = CategoryFrame(scroll, "Motion Primitive Search", "🔍", "#3498db", mp_scripts, self)
        mp_frame.pack(fill="x", padx=10, pady=10)
        
        # Sensitivity Analysis
        sens_scripts = [
            ("Heuristic Weights", "planner/Planner_Sensitivity_Heuristic.py", "Sensitivity - Heuristic Weights"),
            ("True Cost", "planner/Planner_Sensitivity_TrueCost.py", "Sensitivity - True Cost"),
        ]
        
        sens_frame = CategoryFrame(scroll, "Sensitivity Analysis", "📈", "#9b59b6", sens_scripts, self)
        sens_frame.pack(fill="x", padx=10, pady=10)
        
        # Advanced
        adv_scripts = [
            ("Reasoning-based Planner", "planner/reasoning_planner_intersection_scenario.py", "Reasoning Planner"),
            ("Moving Obstacle Avoidance", "planner/moving_obstacle_avoidance.py", "Moving Obstacle Avoidance"),
        ]
        
        adv_frame = CategoryFrame(scroll, "Advanced Planning", "🧠", "#e67e22", adv_scripts, self)
        adv_frame.pack(fill="x", padx=10, pady=10)
    
    def create_controller_tab(self):
        """Create the Controller tab."""
        tab = self.tabview.add("🎮 Controller")
        
        desc = ctk.CTkLabel(
            tab,
            text="Model Predictive Control (MPC) for trajectory tracking",
            font=("Helvetica", 13),
            text_color="gray"
        )
        desc.pack(anchor="w", padx=10, pady=(10, 15))
        
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)
        
        # MPC Scenarios
        mpc_scripts = [
            ("Intersection", "scenarios/mpc_intersection.py", "MPC Intersection"),
            ("Roundabout", "scenarios/mpc_roundabout.py", "MPC Roundabout"),
            ("Multi-lane Intersection", "scenarios/mpc_intersection_multi_lane.py", "MPC Multi-lane"),
            ("Basic Test", "scenarios/mpc_basic.py", "MPC Basic Test"),
        ]
        
        mpc_frame = CategoryFrame(scroll, "MPC Scenarios", "🚦", "#2ecc71", mpc_scripts, self)
        mpc_frame.pack(fill="x", padx=10, pady=10)
        
        # Sensitivity Analysis
        sens_scripts = [
            ("Parameter Sensitivity", "scenarios/mpc_sensitivity_analysis.py", "MPC Sensitivity"),
            ("Cumulative Analysis", "scenarios/mpc_sensitivity_analysis_comulative.py", "Cumulative Sensitivity"),
        ]
        
        sens_frame = CategoryFrame(scroll, "Sensitivity Analysis", "📊", "#e74c3c", sens_scripts, self)
        sens_frame.pack(fill="x", padx=10, pady=10)
        
        # Interactive
        int_scripts = [
            ("Interactive MPC", "scenarios/interactive_mpc.py", "Interactive MPC"),
        ]
        
        int_frame = CategoryFrame(scroll, "Interactive", "🎮", "#1abc9c", int_scripts, self)
        int_frame.pack(fill="x", padx=10, pady=10)
    
    def create_simulation_tab(self):
        """Create the Full Simulation tab."""
        tab = self.tabview.add("📊 Simulation")
        
        desc = ctk.CTkLabel(
            tab,
            text="Complete simulation scenarios with planner and controller",
            font=("Helvetica", 13),
            text_color="gray"
        )
        desc.pack(anchor="w", padx=10, pady=(10, 15))
        
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)
        
        # Scenarios
        sim_scripts = [
            ("Intersection with Obstacles", "scenarios/mpc_intersection.py", "Intersection Simulation"),
            ("Roundabout", "scenarios/mpc_roundabout.py", "Roundabout Simulation"),
            ("Overtaking Cyclist", "scenarios/overtaking_cyclist_bidirectional_road.py", "Cyclist Overtaking"),
        ]
        
        sim_frame = CategoryFrame(scroll, "Full Scenarios", "🚗", "#3498db", sim_scripts, self)
        sim_frame.pack(fill="x", padx=10, pady=10)
        
        # Visualization
        vis_scripts = [
            ("Scenario Visualization", "scenarios/scenario_visualisation.py", "Scenario Visualization"),
        ]
        
        vis_frame = CategoryFrame(scroll, "Visualization", "👁️", "#9b59b6", vis_scripts, self)
        vis_frame.pack(fill="x", padx=10, pady=10)
    
    def create_tools_tab(self):
        """Create the Tools tab."""
        tab = self.tabview.add("🔧 Tools")
        
        desc = ctk.CTkLabel(
            tab,
            text="Utilities and motion primitive generation",
            font=("Helvetica", 13),
            text_color="gray"
        )
        desc.pack(anchor="w", padx=10, pady=(10, 15))
        
        scroll = ctk.CTkScrollableFrame(tab)
        scroll.pack(fill="both", expand=True)
        
        # Motion Primitives
        mp_scripts = [
            ("Generate Bicycle Model MPs", "create_motion_primitives_bicycle_model.py", "Generate Bicycle Model Motion Primitives"),
            ("Generate Prius MPs", "create_motion_primitives_prius.py", "Generate Prius Motion Primitives"),
        ]
        
        mp_frame = CategoryFrame(scroll, "Motion Primitive Generation", "⚙️", "#e67e22", mp_scripts, self)
        mp_frame.pack(fill="x", padx=10, pady=10)
        
        # Info section
        info_frame = ctk.CTkFrame(scroll, corner_radius=15)
        info_frame.pack(fill="x", padx=10, pady=20)
        
        info_title = ctk.CTkLabel(
            info_frame,
            text="ℹ️  About",
            font=("Helvetica", 16, "bold")
        )
        info_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        info_text = ctk.CTkLabel(
            info_frame,
            text="AV Simulation at Intersections\n"
                 "A bi-level framework for modeling 2D vehicular movements\n"
                 "and interactions at urban junctions.\n\n"
                 "Python 3.8-3.11 required | Uses cvxpy, numpy, matplotlib",
            font=("Helvetica", 12),
            text_color="gray",
            justify="left"
        )
        info_text.pack(anchor="w", padx=15, pady=(0, 15))
        
        # Check dependencies button
        check_btn = ctk.CTkButton(
            info_frame,
            text="Check Dependencies",
            command=self.check_dependencies,
            fg_color="#27ae60",
            hover_color="#229954"
        )
        check_btn.pack(pady=(0, 15))
    
    def toggle_theme(self):
        """Toggle between light and dark mode."""
        mode = self.theme_var.get()
        ctk.set_appearance_mode(mode)
    
    def check_dependencies(self):
        """Check if all dependencies are installed."""
        required = ['numpy', 'matplotlib', 'cvxpy', 'tqdm', 'networkx', 'ecos']
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        
        if missing:
            result = messagebox.askyesno(
                "Missing Dependencies",
                f"Missing packages: {', '.join(missing)}\n\nInstall them now?"
            )
            if result:
                subprocess.run([sys.executable, "-m", "pip", "install"] + missing)
                messagebox.showinfo("Success", "Dependencies installed!")
        else:
            messagebox.showinfo("Dependencies", "✓ All dependencies are installed!")


def main():
    """Run the application."""
    app = AVSimulationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
