#!/usr/bin/env python3
"""
AV Simulation at Intersections - Interactive CLI
A modern, lightweight interface for running simulations and analyses.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
MAIN_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(MAIN_DIR))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint
except ImportError:
    print("Installing required packages (rich)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint

console = Console()


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_header():
    """Display the application header."""
    console.print(Panel.fit(
        "[bold blue]🚗 AV Simulation at Intersections[/bold blue]\n"
        "[dim]Motion Planning & Control for Autonomous Vehicles[/dim]",
        border_style="blue"
    ))
    console.print()


def show_main_menu() -> str:
    """Display main menu and get user choice."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Description", style="white")
    
    table.add_row("[1]", "🎯 [bold]Planner[/bold] - Motion primitive search & path planning")
    table.add_row("[2]", "🎮 [bold]Controller[/bold] - MPC-based trajectory tracking")
    table.add_row("[3]", "📊 [bold]Full Simulation[/bold] - Combined planner + controller")
    table.add_row("[q]", "❌ Quit")
    
    console.print(Panel(table, title="[bold]Main Menu[/bold]", border_style="green"))
    
    return Prompt.ask(
        "\n[bold cyan]Select an option[/bold cyan]",
        choices=["1", "2", "3", "q"],
        default="1"
    )


def show_planner_menu() -> str:
    """Display planner submenu."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Description", style="white")
    
    table.add_row("[1]", "🔍 Motion Primitive Search - Intersection")
    table.add_row("[2]", "🔄 Motion Primitive Search - Roundabout")
    table.add_row("[3]", "🛣️  Motion Primitive Search - Multi-lane Intersection")
    table.add_row("[4]", "📈 Sensitivity Analysis - Heuristic Weights")
    table.add_row("[5]", "📊 Sensitivity Analysis - True Cost")
    table.add_row("[6]", "🧠 Reasoning-based Planner")
    table.add_row("[b]", "⬅️  Back to main menu")
    
    console.print(Panel(table, title="[bold]Planner Options[/bold]", border_style="yellow"))
    
    return Prompt.ask(
        "\n[bold cyan]Select an option[/bold cyan]",
        choices=["1", "2", "3", "4", "5", "6", "b"],
        default="1"
    )


def show_controller_menu() -> str:
    """Display controller submenu."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Description", style="white")
    
    table.add_row("[1]", "🚦 MPC Intersection Scenario")
    table.add_row("[2]", "🔄 MPC Roundabout Scenario")
    table.add_row("[3]", "🛣️  MPC Multi-lane Intersection")
    table.add_row("[4]", "🎯 MPC Basic (Simple Test)")
    table.add_row("[5]", "📈 Sensitivity Analysis")
    table.add_row("[6]", "📊 Cumulative Sensitivity Analysis")
    table.add_row("[7]", "🎮 Interactive MPC")
    table.add_row("[b]", "⬅️  Back to main menu")
    
    console.print(Panel(table, title="[bold]Controller (MPC) Options[/bold]", border_style="magenta"))
    
    return Prompt.ask(
        "\n[bold cyan]Select an option[/bold cyan]",
        choices=["1", "2", "3", "4", "5", "6", "7", "b"],
        default="1"
    )


def show_simulation_menu() -> str:
    """Display full simulation submenu."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Option", style="cyan", width=6)
    table.add_column("Description", style="white")
    
    table.add_row("[1]", "🚦 Intersection with Moving Obstacles")
    table.add_row("[2]", "🔄 Roundabout Scenario")
    table.add_row("[3]", "🚴 Overtaking Cyclist (Bidirectional Road)")
    table.add_row("[b]", "⬅️  Back to main menu")
    
    console.print(Panel(table, title="[bold]Full Simulation[/bold]", border_style="blue"))
    
    return Prompt.ask(
        "\n[bold cyan]Select an option[/bold cyan]",
        choices=["1", "2", "3", "b"],
        default="1"
    )


def run_script(script_path: str, description: str):
    """Run a Python script with proper environment setup."""
    full_path = MAIN_DIR / script_path
    
    if not full_path.exists():
        console.print(f"[red]❌ Script not found: {script_path}[/red]")
        return
    
    console.print(f"\n[bold green]▶ Running:[/bold green] {description}")
    console.print(f"[dim]Script: {script_path}[/dim]\n")
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = str(MAIN_DIR)
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(full_path)],
            env=env,
            cwd=str(MAIN_DIR)
        )
        
        if result.returncode == 0:
            console.print("\n[bold green]✓ Completed successfully![/bold green]")
        else:
            console.print(f"\n[bold yellow]⚠ Script exited with code {result.returncode}[/bold yellow]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
    
    console.print()
    Prompt.ask("[dim]Press Enter to continue[/dim]", default="")


def handle_planner_choice(choice: str) -> bool:
    """Handle planner menu selection. Returns True to continue, False to go back."""
    scripts = {
        "1": ("planner/motion_primitive_search.py", "Motion Primitive Search - Intersection"),
        "2": ("planner/motion_primitive_search_roundabout.py", "Motion Primitive Search - Roundabout"),
        "3": ("planner/motion_primitive_search_full_intersection_multi_lane.py", "Motion Primitive Search - Multi-lane"),
        "4": ("planner/Planner_Sensitivity_Heuristic.py", "Sensitivity Analysis - Heuristic Weights"),
        "5": ("planner/Planner_Sensitivity_TrueCost.py", "Sensitivity Analysis - True Cost"),
        "6": ("planner/reasoning_planner_intersection_scenario.py", "Reasoning-based Planner"),
    }
    
    if choice == "b":
        return False
    
    if choice in scripts:
        script, desc = scripts[choice]
        run_script(script, desc)
    
    return True


def handle_controller_choice(choice: str) -> bool:
    """Handle controller menu selection. Returns True to continue, False to go back."""
    scripts = {
        "1": ("scenarios/mpc_intersection.py", "MPC Intersection Scenario"),
        "2": ("scenarios/mpc_roundabout.py", "MPC Roundabout Scenario"),
        "3": ("scenarios/mpc_intersection_multi_lane.py", "MPC Multi-lane Intersection"),
        "4": ("scenarios/mpc_basic.py", "MPC Basic Test"),
        "5": ("scenarios/mpc_sensitivity_analysis.py", "MPC Sensitivity Analysis"),
        "6": ("scenarios/mpc_sensitivity_analysis_comulative.py", "MPC Cumulative Sensitivity Analysis"),
        "7": ("scenarios/interactive_mpc.py", "Interactive MPC"),
    }
    
    if choice == "b":
        return False
    
    if choice in scripts:
        script, desc = scripts[choice]
        run_script(script, desc)
    
    return True


def handle_simulation_choice(choice: str) -> bool:
    """Handle simulation menu selection. Returns True to continue, False to go back."""
    scripts = {
        "1": ("planner/moving_obstacle_avoidance.py", "Intersection with Moving Obstacles"),
        "2": ("scenarios/mpc_roundabout.py", "Roundabout Full Simulation"),
        "3": ("scenarios/overtaking_cyclist_bidirectional_road.py", "Overtaking Cyclist"),
    }
    
    if choice == "b":
        return False
    
    if choice in scripts:
        script, desc = scripts[choice]
        run_script(script, desc)
    
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    console.print("[dim]Checking dependencies...[/dim]")
    
    required = ['numpy', 'matplotlib', 'cvxpy', 'tqdm', 'networkx']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        console.print(f"\n[yellow]⚠ Missing packages: {', '.join(missing)}[/yellow]")
        if Confirm.ask("Install missing packages?", default=True):
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            console.print("[green]✓ Packages installed![/green]")
    else:
        console.print("[green]✓ All dependencies satisfied[/green]")
    
    console.print()


def main():
    """Main application loop."""
    clear_screen()
    show_header()
    check_dependencies()
    
    while True:
        clear_screen()
        show_header()
        
        choice = show_main_menu()
        
        if choice == "q":
            console.print("\n[bold blue]👋 Goodbye![/bold blue]\n")
            break
        
        elif choice == "1":  # Planner
            while True:
                clear_screen()
                show_header()
                planner_choice = show_planner_menu()
                if not handle_planner_choice(planner_choice):
                    break
        
        elif choice == "2":  # Controller
            while True:
                clear_screen()
                show_header()
                controller_choice = show_controller_menu()
                if not handle_controller_choice(controller_choice):
                    break
        
        elif choice == "3":  # Full Simulation
            while True:
                clear_screen()
                show_header()
                sim_choice = show_simulation_menu()
                if not handle_simulation_choice(sim_choice):
                    break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold blue]👋 Goodbye![/bold blue]\n")
        sys.exit(0)
