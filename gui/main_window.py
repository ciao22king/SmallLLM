"""
Main GUI Window for SmallLLM
Complete graphical interface with chat, settings, and model management
Similar to ChatGPT with plugin support
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import Qt first
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox,
        QSlider, QScrollArea, QFrame, QMessageBox, QFileDialog,
        QInputDialog, QProgressBar, QTabWidget, QGroupBox,
        QToolButton, QMenu, QSystemTrayIcon, QSizePolicy
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QTimer, QSize
    from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor, QIcon, QAction
    PYQT_AVAILABLE = True
except ImportError:
    try:
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox,
            QSlider, QScrollArea, QFrame, QMessageBox, QFileDialog,
            QInputDialog, QProgressBar, QTabWidget, QGroupBox,
            QToolButton, QMenu, QSystemTrayIcon, QSizePolicy
        )
        from PySide6.QtCore import Qt, Signal, QObject, QThread, QTimer, QSize
        from PySide6.QtGui import QFont, QTextCursor, QPalette, QColor, QIcon, QAction
        PYQT_AVAILABLE = True
        # Alias for compatibility
        pyqtSignal = Signal
    except ImportError:
        PYQT_AVAILABLE = False
        print("Error: PyQt6 or PySide6 is required. Install with: pip install PyQt6 or PySide6")
        sys.exit(1)

# Import SmallLLM components
try:
    from smallllm import SmallLLM, Tokenizer, ModelConfig, get_config_by_name
    from smallllm.config import TINY_CONFIG, SMALL_CONFIG, MEDIUM_CONFIG
    SMALLLLM_AVAILABLE = True
except ImportError:
    SMALLLLM_AVAILABLE = False
    print("Warning: SmallLLM not available. Some features will be disabled.")

# Import GUI components
try:
    from .plugin_manager import PluginManager
    from .plugins import (
        CodeCompleterPlugin, VSCodeIntegrationPlugin, WebSearchPlugin,
        MathSolverPlugin, TranslatorPlugin
    )
    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False
    print("Warning: Plugins not available. Plugin features will be disabled.")


class SignalEmitter(QObject):
    """Helper class for cross-thread signals."""
    update_chat = pyqtSignal(str, str)  # (message, sender)
    update_status = pyqtSignal(str)
    update_progress = pyqtSignal(int)
    model_loaded = pyqtSignal(bool)
    generation_complete = pyqtSignal(str)


class GenerationThread(QThread):
    """Thread for text generation to avoid UI freezing."""
    
    def __init__(self, model, prompt, max_tokens, temperature, top_k, do_sample, plugin_manager=None, context=None):
        super().__init__()
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.do_sample = do_sample
        self.plugin_manager = plugin_manager
        self.context = context or {}
        self.result = ""
        self.error = None
    
    def run(self):
        try:
            # Apply pre-generation plugins
            if self.plugin_manager:
                self.prompt = self.plugin_manager.pre_generate(self.prompt, self.context)
            
            self.result = self.model.generate(
                prompt=self.prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                do_sample=self.do_sample
            )
            
            # Apply post-generation plugins
            if self.plugin_manager:
                self.result = self.plugin_manager.post_generate(self.result, self.context)
        except Exception as e:
            self.error = str(e)
    
    def get_result(self):
        return self.result, self.error


class ChatMessageWidget(QWidget):
    """Widget for displaying a chat message with ChatGPT-like styling."""
    
    def __init__(self, text: str, sender: str = "user"):
        super().__init__()
        self.sender = sender
        
        # Main layout
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # Avatar
        self.avatar = QLabel()
        self.avatar.setFixedSize(32, 32)
        self.avatar.setStyleSheet("border-radius: 16px; background: #6366f1; color: white;")
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if sender == "user":
            self.avatar.setText("\u0001F464")
            self.setStyleSheet("""
                background: #6366f1;
                color: white;
                border-radius: 12px;
                padding: 8px;
            """)
        else:
            self.avatar.setText("\u0001F916")
            self.setStyleSheet("""
                background: #444654;
                color: white;
                border-radius: 12px;
                padding: 8px;
            """)
        
        # Message content
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            border: none;
            background: transparent;
            color: inherit;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
        """)
        self.text_edit.setFont(QFont("Segoe UI", 10))
        self.text_edit.setText(text)
        self.text_edit.setSizePolicy(
            QWidget.SizePolicy.Policy.Expanding,
            QWidget.SizePolicy.Policy.Minimum
        )
        
        # Adjust layout based on sender
        if sender == "user":
            layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignRight)
            layout.addWidget(self.text_edit, 1, Qt.AlignmentFlag.AlignRight)
        else:
            layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignLeft)
            layout.addWidget(self.text_edit, 1, Qt.AlignmentFlag.AlignLeft)
        
        self.setLayout(layout)


class ChatWidget(QWidget):
    """Chat widget with message history and ChatGPT-like interface."""
    
    def __init__(self):
        super().__init__()
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: #343541;")
        
        # Container for messages
        self.messages_container = QWidget()
        self.messages_container.setStyleSheet("background: #343541;")
        self.messages_layout = QVBoxLayout()
        self.messages_layout.setContentsMargins(10, 10, 10, 10)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch()
        self.messages_container.setLayout(self.messages_layout)
        
        self.scroll_area.setWidget(self.messages_container)
        layout.addWidget(self.scroll_area, 1)
        
        self.setLayout(layout)
    
    def add_message(self, text: str, sender: str = "user"):
        """Add a message to the chat."""
        message_widget = ChatMessageWidget(text, sender)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, message_widget)
        
        # Auto-scroll to bottom
        QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))
    
    def clear(self):
        """Clear all messages."""
        for i in reversed(range(self.messages_layout.count())):
            widget = self.messages_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.messages_layout.addStretch()


class PluginButton(QToolButton):
    """Button for plugin actions."""
    
    def __init__(self, plugin_name: str, display_name: str, icon: str, parent=None):
        super().__init__(parent)
        self.plugin_name = plugin_name
        self.setText(f"{icon} {display_name}")
        self.setStyleSheet("""
            QToolButton {
                background: #444654;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QToolButton:hover {
                background: #4e505e;
            }
            QToolButton:checked {
                background: #6366f1;
            }
        """)
        self.setCheckable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class SettingsWidget(QWidget):
    """Settings panel for the GUI with advanced options."""
    
    def __init__(self, plugin_manager=None):
        super().__init__()
        self.plugin_manager = plugin_manager
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Model Configuration
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "small", "medium"])
        self.model_combo.setCurrentText("small")
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Generation Settings
        gen_group = QGroupBox("Generation Settings")
        gen_layout = QVBoxLayout()
        
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)  # x10 for precision
        self.temp_slider.setValue(7)  # 0.7
        self.temp_slider.setTickInterval(1)
        self.temp_label = QLabel("Temperature: 0.7")
        self.temp_slider.valueChanged.connect(self.update_temp_label)
        gen_layout.addWidget(self.temp_label)
        gen_layout.addWidget(self.temp_slider)
        
        self.tokens_slider = QSlider(Qt.Orientation.Horizontal)
        self.tokens_slider.setRange(10, 500)
        self.tokens_slider.setValue(50)
        self.tokens_slider.setTickInterval(10)
        self.tokens_label = QLabel("Max Tokens: 50")
        self.tokens_slider.valueChanged.connect(self.update_tokens_label)
        gen_layout.addWidget(self.tokens_label)
        gen_layout.addWidget(self.tokens_slider)
        
        self.topk_slider = QSlider(Qt.Orientation.Horizontal)
        self.topk_slider.setRange(10, 100)
        self.topk_slider.setValue(50)
        self.topk_slider.setTickInterval(10)
        self.topk_label = QLabel("Top-k: 50")
        self.topk_slider.valueChanged.connect(self.update_topk_label)
        gen_layout.addWidget(self.topk_label)
        gen_layout.addWidget(self.topk_slider)
        
        self.sample_checkbox = QPushButton("Sampling: ON")
        self.sample_checkbox.setCheckable(True)
        self.sample_checkbox.setChecked(True)
        self.sample_checkbox.setStyleSheet("QPushButton:checked { background: #6366f1; }")
        gen_layout.addWidget(self.sample_checkbox)
        
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)
        
        # Plugin Settings
        if PLUGINS_AVAILABLE:
            plugin_group = QGroupBox("Plugin Settings")
            plugin_layout = QVBoxLayout()
            
            self.plugin_buttons = {}
            
            # Create plugin buttons
            if self.plugin_manager:
                for plugin_spec in self.plugin_manager.get_plugin_list():
                    btn = PluginButton(
                        plugin_spec.name,
                        plugin_spec.display_name,
                        plugin_spec.icon
                    )
                    btn.setChecked(plugin_spec.enabled)
                    btn.toggled.connect(lambda checked, name=plugin_spec.name: 
                                       self.on_plugin_toggled(name, checked))
                    plugin_layout.addWidget(btn)
                    self.plugin_buttons[plugin_spec.name] = btn
            
            plugin_group.setLayout(plugin_layout)
            layout.addWidget(plugin_group)
        
        # Model Actions
        actions_group = QGroupBox("Model Actions")
        actions_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.setStyleSheet("background: #6366f1; color: white; border: none; padding: 8px;")
        actions_layout.addWidget(self.load_btn)
        
        self.save_btn = QPushButton("Save Model")
        self.save_btn.setStyleSheet("background: #374151; color: white; border: none; padding: 8px;")
        self.save_btn.setEnabled(False)
        actions_layout.addWidget(self.save_btn)
        
        self.quantize_btn = QPushButton("Quantize (8-bit)")
        self.quantize_btn.setStyleSheet("background: #374151; color: white; border: none; padding: 8px;")
        self.quantize_btn.setEnabled(False)
        actions_layout.addWidget(self.quantize_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #60a5fa;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def on_plugin_toggled(self, plugin_name: str, checked: bool):
        """Handle plugin toggle."""
        if self.plugin_manager:
            self.plugin_manager.enable_plugin(plugin_name, checked)
    
    def update_temp_label(self, value):
        temp = value / 10
        self.temp_label.setText(f"Temperature: {temp:.1f}")
    
    def update_tokens_label(self, value):
        self.tokens_label.setText(f"Max Tokens: {value}")
    
    def update_topk_label(self, value):
        self.topk_label.setText(f"Top-k: {value}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            "model": self.model_combo.currentText(),
            "temperature": self.temp_slider.value() / 10,
            "max_tokens": self.tokens_slider.value(),
            "top_k": self.topk_slider.value(),
            "do_sample": self.sample_checkbox.isChecked()
        }


class PluginPanel(QWidget):
    """Panel for managing and configuring plugins."""
    
    def __init__(self, plugin_manager: PluginManager):
        super().__init__()
        self.plugin_manager = plugin_manager
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Plugin Manager")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        # Plugin list
        self.plugin_widgets = {}
        
        for plugin_spec in self.plugin_manager.get_plugin_list():
            widget = self.create_plugin_widget(plugin_spec)
            layout.addWidget(widget)
            self.plugin_widgets[plugin_spec.name] = widget
        
        layout.addStretch()
        self.setLayout(layout)
    
    def create_plugin_widget(self, plugin_spec) -> QWidget:
        """Create a widget for a plugin."""
        widget = QWidget()
        widget.setStyleSheet("background: #25262b; border-radius: 8px; padding: 10px;")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Icon and name
        icon_label = QLabel(plugin_spec.icon)
        icon_label.setStyleSheet("font-size: 20px;")
        icon_label.setFixedWidth(30)
        
        name_label = QLabel(plugin_spec.display_name)
        name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")
        
        desc_label = QLabel(plugin_spec.description)
        desc_label.setStyleSheet("font-size: 12px; color: #8b949e;")
        
        # Toggle button
        toggle_btn = QPushButton("ON" if plugin_spec.enabled else "OFF")
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(plugin_spec.enabled)
        toggle_btn.setStyleSheet("""
            QPushButton:checked { background: #10b981; color: white; }
            QPushButton { background: #374151; color: white; }
        """)
        toggle_btn.toggled.connect(lambda checked, name=plugin_spec.name: 
                                   self.on_plugin_toggled(name, checked))
        
        # Configure button
        config_btn = QPushButton("\u0001F52C")
        config_btn.setStyleSheet("background: #444654; color: white;")
        config_btn.clicked.connect(lambda: self.configure_plugin(plugin_spec.name))
        
        layout.addWidget(icon_label)
        layout.addWidget(name_label, 1)
        layout.addWidget(desc_label, 2)
        layout.addWidget(toggle_btn)
        layout.addWidget(config_btn)
        
        widget.setLayout(layout)
        return widget
    
    def on_plugin_toggled(self, plugin_name: str, checked: bool):
        """Handle plugin toggle."""
        self.plugin_manager.enable_plugin(plugin_name, checked)
    
    def configure_plugin(self, plugin_name: str):
        """Configure a plugin."""
        plugin = self.plugin_manager.load_plugin(plugin_name)
        if plugin and hasattr(plugin, 'get_config_schema'):
            schema = plugin.get_config_schema()
            # Show configuration dialog
            self.show_config_dialog(plugin_name, schema)
    
    def show_config_dialog(self, plugin_name: str, schema: Dict[str, Any]):
        """Show configuration dialog for a plugin."""
        # Simple implementation - just show the schema
        info = QMessageBox(self)
        info.setWindowTitle(f"Configure {plugin_name}")
        info.setText(f"Configuration options:\n\n{json.dumps(schema, indent=2)}")
        info.exec()


class MainWindow(QMainWindow):
    """Main GUI window for SmallLLM with ChatGPT-like interface."""
    
    def __init__(self):
        super().__init__()
        
        # Window settings
        self.setWindowTitle("SmallLLM - Tiny AI for Old Hardware")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(800, 600)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Initialize model
        self.model: Optional[SmallLLM] = None
        self.signal_emitter = SignalEmitter()
        
        # Initialize plugin manager
        self.plugin_manager = None
        if PLUGINS_AVAILABLE:
            self.plugin_manager = PluginManager()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Create sidebar (for future use)
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        self.sidebar.setStyleSheet("background: #202123; border-right: 1px solid #2f3032;")
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)
        
        # New chat button
        new_chat_btn = QPushButton("\u0001F4AC New Chat")
        new_chat_btn.setStyleSheet("""
            QPushButton {
                background: #444654;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                text-align: left;
                font-size: 14px;
                margin: 8px;
            }
            QPushButton:hover {
                background: #4e505e;
            }
        """)
        new_chat_btn.clicked.connect(self.new_chat)
        sidebar_layout.addWidget(new_chat_btn)
        
        # History (placeholder)
        history_label = QLabel("History")
        history_label.setStyleSheet("color: #8b949e; margin-left: 8px; margin-top: 8px;")
        sidebar_layout.addWidget(history_label)
        
        # Plugin menu button
        plugin_btn = QPushButton("\u0001F916 Plugins")
        plugin_btn.setStyleSheet("""
            QPushButton {
                background: #444654;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                text-align: left;
                font-size: 14px;
                margin: 8px;
            }
            QPushButton:hover {
                background: #4e505e;
            }
        """)
        plugin_btn.clicked.connect(self.show_plugin_panel)
        sidebar_layout.addWidget(plugin_btn)
        
        sidebar_layout.addStretch()
        self.sidebar.setLayout(sidebar_layout)
        main_layout.addWidget(self.sidebar)
        
        # Create main content area
        content_widget = QWidget()
        content_widget.setStyleSheet("background: #343541;")
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab { 
                background: #25262b; 
                color: #8b949e; 
                padding: 8px 16px; 
                border: none;
            }
            QTabBar::tab:selected { 
                background: #1e1e1e; 
                color: white; 
                border-bottom: 2px solid #6366f1;
            }
            QTabBar::tab:hover { 
                background: #2b2c2f; 
            }
        """)
        content_layout.addWidget(self.tab_widget)
        content_widget.setLayout(content_layout)
        main_layout.addWidget(content_widget, 1)
        
        # Create tabs
        self.create_chat_tab()
        self.create_settings_tab()
        self.create_info_tab()
        self.create_plugin_tab()
        
        # Connect signals
        self.connect_signals()
        
        # Load model
        self.load_model("small")
    
    def set_dark_theme(self):
        """Set dark theme for the application."""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(99, 102, 241))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(99, 102, 241))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        dark_palette.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.Text, Qt.GlobalColor.gray)
        dark_palette.setColor(QPalette.ColorRole.Disabled, QPalette.ColorRole.ButtonText, Qt.GlobalColor.gray)
        
        self.setPalette(dark_palette)
    
    def create_chat_tab(self):
        """Create the chat tab with ChatGPT-like interface."""
        chat_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Chat widget
        self.chat_widget = ChatWidget()
        layout.addWidget(self.chat_widget, 1)
        
        # Input area
        input_widget = QWidget()
        input_widget.setStyleSheet("background: #444654; border-top: 1px solid #37383e;")
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(15, 10, 15, 10)
        input_layout.setSpacing(10)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.setStyleSheet("""
            QLineEdit { 
                background: #25262b; 
                border: 1px solid #37383e; 
                border-radius: 8px; 
                color: white; 
                padding: 10px 14px;
                font-size: 14px;
            }
        """)
        self.input_field.setFont(QFont("Segoe UI", 10))
        self.input_field.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("\u0001F300")
        self.send_btn.setStyleSheet("""
            QPushButton { 
                background: #6366f1; 
                color: white; 
                border: none; 
                border-radius: 8px; 
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover { background: #5558e3; }
            QPushButton:pressed { background: #4749d1; }
            QPushButton:disabled { background: #37383e; color: #6e7681; }
        """)
        self.send_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setEnabled(False)
        
        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.send_btn)
        input_widget.setLayout(input_layout)
        layout.addWidget(input_widget)
        
        chat_tab.setLayout(layout)
        self.tab_widget.addTab(chat_tab, "\u0001F4AC Chat")
        
        # Add welcome message
        self.chat_widget.add_message(
            "Ciao! Sono SmallLLM, la tua AI locale.\n\n"
            "Posso aiutarti con:\n"
            "- Risposte a domande\n"
            "- Generazione di testo\n"
            "- Chat interattiva\n"
            "- Plugin per funzionalita avanzate\n\n"
            "Prova a scrivermi qualcosa!",
            "ai"
        )
    
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.settings_widget = SettingsWidget(self.plugin_manager)
        layout.addWidget(self.settings_widget)
        
        settings_tab.setLayout(layout)
        self.tab_widget.addTab(settings_tab, "\u0001F52C Settings")
        
        # Connect buttons
        self.settings_widget.load_btn.clicked.connect(self.load_model_from_settings)
        self.settings_widget.save_btn.clicked.connect(self.save_model)
        self.settings_widget.quantize_btn.clicked.connect(self.quantize_model)
    
    def create_info_tab(self):
        """Create the info tab."""
        info_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("""
            QTextEdit { 
                background: #1e1e1e; 
                border: 1px solid #37383e; 
                border-radius: 6px; 
                color: #8b949e;
            }
        """)
        self.info_text.setFont(QFont("Consolas", 10))
        
        info_layout.addWidget(self.info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Memory info
        memory_group = QGroupBox("Memory Usage")
        memory_layout = QVBoxLayout()
        
        self.memory_text = QTextEdit()
        self.memory_text.setReadOnly(True)
        self.memory_text.setStyleSheet("""
            QTextEdit { 
                background: #1e1e1e; 
                border: 1px solid #37383e; 
                border-radius: 6px; 
                color: #8b949e;
            }
        """)
        self.memory_text.setFont(QFont("Consolas", 10))
        
        memory_layout.addWidget(self.memory_text)
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        info_tab.setLayout(layout)
        self.tab_widget.addTab(info_tab, "\u0001F19D Info")
    
    def create_plugin_tab(self):
        """Create the plugin management tab."""
        if not PLUGINS_AVAILABLE or not self.plugin_manager:
            return
        
        plugin_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.plugin_panel = PluginPanel(self.plugin_manager)
        layout.addWidget(self.plugin_panel)
        
        plugin_tab.setLayout(layout)
        self.tab_widget.addTab(plugin_tab, "\u0001F916 Plugins")
    
    def connect_signals(self):
        """Connect all signals."""
        pass
    
    def new_chat(self):
        """Start a new chat."""
        self.chat_widget.clear()
        self.chat_widget.add_message(
            "Ciao! Sono SmallLLM, la tua AI locale.\n\n"
            "Come posso aiutarti?",
            "ai"
        )
    
    def show_plugin_panel(self):
        """Show the plugin panel."""
        # Switch to plugin tab
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "\u0001F916 Plugins":
                self.tab_widget.setCurrentIndex(i)
                break
    
    def load_model(self, config_name: str):
        """Load a model with the given configuration."""
        try:
            if self.settings_widget:
                self.settings_widget.status_label.setText("Status: Loading model...")
                self.settings_widget.status_label.setStyleSheet("color: #fbbf24;")
            QApplication.processEvents()
            
            config = get_config_by_name(config_name)
            self.model = SmallLLM(config)
            self.model._ensure_initialized()
            
            if self.settings_widget:
                self.settings_widget.status_label.setText("Status: Model loaded successfully!")
                self.settings_widget.status_label.setStyleSheet("color: #10b981;")
            self.send_btn.setEnabled(True)
            if self.settings_widget:
                self.settings_widget.save_btn.setEnabled(True)
                self.settings_widget.quantize_btn.setEnabled(True)
            
            # Update info
            self.update_info()
            
        except Exception as e:
            if self.settings_widget:
                self.settings_widget.status_label.setText(f"Status: Error - {str(e)}")
                self.settings_widget.status_label.setStyleSheet("color: #ef4444;")
            self.send_btn.setEnabled(False)
    
    def load_model_from_settings(self):
        """Load model based on settings."""
        config_name = self.settings_widget.model_combo.currentText()
        self.load_model(config_name)
    
    def update_info(self):
        """Update model information display."""
        if self.model is None or not self.info_text:
            return
        
        config = self.model.config
        config_name = self.settings_widget.model_combo.currentText() if self.settings_widget else "small"
        
        info_text = f"""
Model: SmallLLM ({config_name})

Architecture:
  Vocabulary Size: {config.vocab_size}
  Embedding Dim: {config.embedding_dim}
  Hidden Dim: {config.hidden_dim}
  Layers: {config.num_layers}
  Heads: {config.num_heads}
  Head Dim: {config.head_dim}
  Max Seq Len: {config.max_seq_len}

Parameters:
  Estimated: {config.num_params:,}

Generation:
  Temperature: {self.settings_widget.temp_slider.value() / 10:.1f if self.settings_widget else 0.7}
  Max Tokens: {self.settings_widget.tokens_slider.value() if self.settings_widget else 50}
  Top-k: {self.settings_widget.topk_slider.value() if self.settings_widget else 50}

Lazy Loading:
  Parameters are loaded on-demand to save memory
  """
        self.info_text.setText(info_text)
        
        # Memory info
        if hasattr(self.model, 'get_memory_usage'):
            memory = self.model.get_memory_usage()
            memory_text = "Memory Usage:\n"
            for key, value in memory.items():
                memory_text += f"  {key}: {value / 1024 / 1024:.2f} MB\n"
            self.memory_text.setText(memory_text)
    
    def send_message(self):
        """Send a message to the AI."""
        text = self.input_field.text().strip()
        if not text or self.model is None:
            return
        
        # Add user message
        self.chat_widget.add_message(text, "user")
        self.input_field.clear()
        self.input_field.setFocus()
        
        # Add typing indicator
        self.chat_widget.add_message("Thinking...", "ai")
        
        # Get config from settings
        config = self.settings_widget.get_config() if self.settings_widget else {
            "max_tokens": 50,
            "temperature": 0.7,
            "top_k": 50,
            "do_sample": True
        }
        
        # Create context with plugin info
        context = {
            "model": self.model.config.name if hasattr(self.model.config, 'name') else "small",
            "plugins_enabled": []
        }
        
        if self.plugin_manager:
            context["plugins_enabled"] = [
                p.name for p in self.plugin_manager.get_enabled_plugins()
            ]
        
        # Start generation in a thread
        self.thread = GenerationThread(
            self.model,
            text,
            config["max_tokens"],
            config["temperature"],
            config["top_k"],
            config["do_sample"],
            self.plugin_manager,
            context
        )
        self.thread.finished.connect(self.on_generation_complete)
        self.thread.start()
        
        # Disable input while generating
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
    
    def on_generation_complete(self):
        """Handle completion of text generation."""
        result, error = self.thread.get_result()
        
        # Remove typing indicator
        if self.chat_widget.messages_layout.count() > 0:
            last_widget = self.chat_widget.messages_layout.itemAt(
                self.chat_widget.messages_layout.count() - 2
            ).widget()
            if last_widget and "Thinking" in last_widget.text_edit.toPlainText():
                last_widget.deleteLater()
        
        # Add AI response
        if error:
            self.chat_widget.add_message(f"Error: {error}", "ai")
        else:
            self.chat_widget.add_message(result, "ai")
        
        # Re-enable input
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
    
    def save_model(self):
        """Save the current model."""
        if self.model is None:
            return
        
        path = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Model"
        )
        if path:
            try:
                self.model.save_weights(path)
                QMessageBox.information(
                    self, "Success", f"Model saved to {path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to save model: {str(e)}"
                )
    
    def quantize_model(self):
        """Quantize the model to 8-bit."""
        if self.model is None:
            return
        
        try:
            self.model.quantize()
            QMessageBox.information(
                self, "Success", "Model quantized to 8-bit!"
            )
            self.update_info()
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to quantize model: {str(e)}"
            )
    
    def closeEvent(self, event):
        """Handle window close event."""
        event.accept()


def main():
    """Main entry point for the GUI."""
    app = QApplication(sys.argv)
    app.setApplicationName("SmallLLM")
    app.setOrganizationName("ciao22king")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
