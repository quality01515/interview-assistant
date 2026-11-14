"""
License dialog: shows this computer's machine ID (with a Copy button) so the
user can request a key, and accepts a key that is checked against this machine.

Two uses:
  - required=True  — at startup when the app may not run (trial over, license
    expired, old key retired, clock set back). Quit exits the app.
  - required=False — from the setup screen's "License…" button, to enter a key
    early or copy the machine ID. Cancel just closes it.
"""
from __future__ import annotations

import re
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QVBoxLayout,
)

from pipeline.license import (
    R_CLOCK, R_INVALID, R_LEGACY_RETIRED, LicenseError, LicenseInfo,
    check_key, describe, is_legacy_key,
)
from pipeline.machine_id import machine_id, machine_id_bytes
from ui.style import STYLE


class LicenseDialog(QDialog):
    """
    After exec() == Accepted: `key` holds the validated key and `info` its
    decoded contents — except in the clock case, where Accepted means "Retry"
    and `key` is empty (the caller re-checks after the user fixed the clock).
    """

    def __init__(
        self,
        *,
        reason: str = "",
        info: Optional[LicenseInfo] = None,
        last_seen: float = 0.0,
        status_text: str = "",
        required: bool = True,
    ):
        super().__init__()
        self._clock_mode = reason == R_CLOCK
        self.setWindowTitle(
            "Check your computer's date" if self._clock_mode
            else "License key required" if required else "License"
        )
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(600)
        if required:
            # Not closable via X — must click Quit (or Activate).
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        self.key: str = ""
        self.info: Optional[LicenseInfo] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        message = describe(reason, info, last_seen) if reason else status_text
        if message:
            body = QLabel(message)
            body.setWordWrap(True)
            body.setStyleSheet("font-size: 14px;" + (" font-weight: bold;" if required else ""))
            layout.addWidget(body)

        # ── machine ID ──
        mid = machine_id()
        hint = QLabel(
            "Your machine ID — send it to get a license key for this computer:"
            if mid else "This computer's machine ID could not be read."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        if mid:
            mid_row = QHBoxLayout()
            self.mid_field = QLineEdit(mid)
            self.mid_field.setReadOnly(True)
            self.mid_field.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size: 15px;")
            self.btn_copy = QPushButton("Copy")
            self.btn_copy.clicked.connect(lambda: self._copy(mid))
            mid_row.addWidget(self.mid_field, stretch=1)
            mid_row.addWidget(self.btn_copy)
            layout.addLayout(mid_row)

        # ── key entry ──
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Paste your license key (starts with AS2-)")
        self.input.setFixedHeight(80)
        self.input.textChanged.connect(self._clear_error)
        if self._clock_mode:
            self.input.setVisible(False)
        layout.addWidget(self.input)

        self.lbl_error = QLabel("")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setStyleSheet("color: #f38ba8; font-weight: bold;")
        layout.addWidget(self.lbl_error)

        btn_row = QHBoxLayout()
        self.btn_quit = QPushButton("Quit" if required else "Cancel")
        if required:
            self.btn_quit.setObjectName("danger")
        self.btn_quit.clicked.connect(self.reject)
        self.btn_activate = QPushButton("Retry" if self._clock_mode else "Activate")
        self.btn_activate.setObjectName("primary")
        self.btn_activate.setDefault(True)
        self.btn_activate.clicked.connect(self._activate)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_quit)
        btn_row.addWidget(self.btn_activate)
        layout.addLayout(btn_row)

    def _copy(self, mid: str) -> None:
        QApplication.clipboard().setText(mid)
        self.btn_copy.setText("Copied")
        QTimer.singleShot(1500, lambda: self.btn_copy.setText("Copy"))

    def _clear_error(self) -> None:
        self.lbl_error.setText("")

    def _activate(self) -> None:
        if self._clock_mode:
            self.accept()
            return
        candidate = re.sub(r"\s+", "", self.input.toPlainText())
        if not candidate:
            self.lbl_error.setText("Paste a license key first.")
            return
        if is_legacy_key(candidate):
            self.lbl_error.setText(describe(R_LEGACY_RETIRED))
            return
        try:
            machine = machine_id_bytes()
        except Exception:
            self.lbl_error.setText("This computer's machine ID could not be read.")
            return
        try:
            info = check_key(candidate, machine)
        except LicenseError as e:
            self.lbl_error.setText(describe(e.reason or R_INVALID, e.info))
            return
        self.key = candidate
        self.info = info
        self.accept()
