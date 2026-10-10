"""
Styles for SmallLLM GUI
"""

CHATGPT_STYLESHEET = """
/* Main Window */
QMainWindow {
    background-color: #1e1e1e;
}

/* Chat Area */
QWidget#ChatArea {
    background-color: #1e1e1e;
}

/* Messages */
ChatMessageWidget {
    margin: 8px;
    padding: 12px 16px;
    border-radius: 12px;
}

ChatMessageWidget[user="true"] {
    background-color: #6366f1;
    margin-left: 20%;
    border-bottom-right-radius: 4px;
}

ChatMessageWidget[user="false"] {
    background-color: #25262b;
    margin-right: 20%;
    border-bottom-left-radius: 4px;
}

/* Input Area */
QWidget#InputArea {
    background-color: #25262b;
    border-top: 1px solid #37383e;
    padding: 12px;
}

QLineEdit#InputField {
    background-color: #1e1e1e;
    border: 1px solid #37383e;
    border-radius: 8px;
    color: #ffffff;
    padding: 10px 14px;
    font-size: 14px;
}

QLineEdit#InputField:focus {
    border-color: #6366f1;
    outline: none;
}

QPushButton#SendButton {
    background-color: #6366f1;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    font-size: 14px;
}

QPushButton#SendButton:hover {
    background-color: #5558e3;
}

QPushButton#SendButton:pressed {
    background-color: #4749d1;
}

QPushButton#SendButton:disabled {
    background-color: #37383e;
    color: #6e7681;
}

/* Sidebar */
QWidget#Sidebar {
    background-color: #202123;
    border-right: 1px solid #2f3032;
}

QPushButton#NewChatButton {
    background-color: #444654;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px;
    text-align: left;
    font-size: 14px;
}

QPushButton#NewChatButton:hover {
    background-color: #4e505e;
}

/* History Items */
HistoryItem {
    padding: 10px;
    border-radius: 8px;
    color: #d1d5db;
}

HistoryItem:hover {
    background-color: #2b2c2f;
}

HistoryItem[active="true"] {
    background-color: #2b2c2f;
    color: white;
}

/* Settings */
QComboBox {
    background-color: #25262b;
    color: white;
    border: 1px solid #37383e;
    border-radius: 6px;
    padding: 8px;
    min-width: 150px;
}

QComboBox QAbstractItemView {
    background-color: #25262b;
    color: white;
    selection-background-color: #6366f1;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #37383e;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    background: #6366f1;
    border-radius: 8px;
    margin: -6px 0;
}

QSlider::handle:horizontal:hover {
    background: #5558e3;
}

/* Tabs */
QTabWidget::pane {
    border: none;
    background-color: #1e1e1e;
}

QTabBar::tab {
    background-color: #25262b;
    color: #8b949e;
    padding: 10px 16px;
    border: none;
    border-radius: 0;
}

QTabBar::tab:selected {
    background-color: #1e1e1e;
    color: white;
    border-bottom: 2px solid #6366f1;
}

QTabBar::tab:hover {
    background-color: #2b2c2f;
}

/* Scrollbars */
QScrollBar:vertical {
    width: 8px;
    background: transparent;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #40414f;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #4e505e;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* Group Boxes */
QGroupBox {
    border: 1px solid #37383e;
    border-radius: 8px;
    margin-top: 8px;
    padding: 12px;
    color: #8b949e;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #8b949e;
}

/* Buttons */
QPushButton {
    background-color: #37383e;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
}

QPushButton:hover {
    background-color: #4e505e;
}

QPushButton:pressed {
    background-color: #2b2c2f;
}

/* Labels */
QLabel {
    color: #d1d5db;
    font-size: 14px;
}

QLabel#StatusLabel {
    color: #60a5fa;
}

/* Text Edit */
QTextEdit {
    background-color: #1e1e1e;
    color: #d1d5db;
    border: 1px solid #37383e;
    border-radius: 6px;
    padding: 10px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
}

QTextEdit:read-only {
    border: none;
    background-color: transparent;
}

/* Message Box */
QMessageBox {
    background-color: #25262b;
    color: white;
}

QMessageBox QLabel {
    color: #d1d5db;
}

QMessageBox QPushButton {
    background-color: #37383e;
    color: white;
    min-width: 80px;
}

/* Tooltips */
QToolTip {
    background-color: #25262b;
    color: white;
    border: 1px solid #37383e;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

/* Menu */
QMenu {
    background-color: #25262b;
    color: white;
    border: 1px solid #37383e;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 8px 12px;
    color: #d1d5db;
}

QMenu::item:selected {
    background-color: #6366f1;
    color: white;
}

QMenu::item:disabled {
    color: #4e505e;
}

/* Status Bar */
QStatusBar {
    background-color: #25262b;
    color: #8b949e;
    border-top: 1px solid #37383e;
}

/* Title Bar */
QMainWindow::title {
    background-color: #202123;
}
"""

DARK_STYLESHEET = CHATGPT_STYLESHEET

LIGHT_STYLESHEET = """
/* Light Theme */
QMainWindow {
    background-color: #ffffff;
}

QWidget#ChatArea {
    background-color: #f7f7f8;
}

ChatMessageWidget {
    margin: 8px;
    padding: 12px 16px;
    border-radius: 12px;
}

ChatMessageWidget[user="true"] {
    background-color: #6366f1;
    color: white;
    margin-left: 20%;
    border-bottom-right-radius: 4px;
}

ChatMessageWidget[user="false"] {
    background-color: #ffffff;
    color: #1f2937;
    margin-right: 20%;
    border-bottom-left-radius: 4px;
    border: 1px solid #e5e7eb;
}

QWidget#InputArea {
    background-color: #ffffff;
    border-top: 1px solid #e5e7eb;
    padding: 12px;
}

QLineEdit#InputField {
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    color: #1f2937;
    padding: 10px 14px;
    font-size: 14px;
}

QPushButton#SendButton {
    background-color: #6366f1;
    color: white;
}

QWidget#Sidebar {
    background-color: #f7f7f8;
    border-right: 1px solid #e5e7eb;
}

QComboBox {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
}

QSlider::groove:horizontal {
    background: #d1d5db;
}

QSlider::handle:horizontal {
    background: #6366f1;
}

QTabWidget::pane {
    background-color: #f7f7f8;
}

QTabBar::tab {
    background-color: #f7f7f8;
    color: #6b7280;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1f2937;
    border-bottom: 2px solid #6366f1;
}

QScrollBar:vertical {
    background: transparent;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
}

QGroupBox {
    border: 1px solid #e5e7eb;
    color: #6b7280;
}

QPushButton {
    background-color: #f3f4f6;
    color: #1f2937;
}

QLabel {
    color: #374151;
}
"""


def apply_stylesheet(app, theme="dark"):
    """Apply stylesheet to the application."""
    if theme == "dark":
        app.setStyleSheet(CHATGPT_STYLESHEET)
    else:
        app.setStyleSheet(LIGHT_STYLESHEET)
