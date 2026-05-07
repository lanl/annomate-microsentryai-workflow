# MVC Architecture — AnnoMate & MicroSentryAI

This project uses Qt's Model/View architecture with an explicit Controller layer. Each layer has a strict boundary: `core/` is pure Python with zero Qt dependencies, `models/` are Qt adapter classes that wrap core states, `controllers/` are headless QObject services that own business logic, and `views/` are Qt widget trees that display data and express user intent.

---

## Layer Boundaries

**`core/` — Pure Python, zero Qt dependencies**
- States, persistence, logic, and utilities import nothing from Qt.
- Fully testable with plain `pytest` — no `QApplication` required.
- Domain colors are `(r, g, b)` tuples; geometry is plain Python lists/tuples.

**`models/` — Qt Model adapters**
- Subclass `QAbstractTableModel` (or similar). Wrap one `core/states/` object each.
- Expose a **query API** (read-only, no side effects) and a **command API** (mutate → emit signal).
- Emit `dataChanged` or `modelReset` after every mutation. Views poll nothing; they react to signals.
- Only Qt infrastructure imports allowed: `QAbstractTableModel`, `QModelIndex`, `Signal`, etc.
- No business logic. Input validation belongs in `core/` or controllers.

**`controllers/` — Headless services**
- `QObject`-based; may use `QThread` / `QRunnable` for background work.
- No widget types whatsoever (`QWidget`, `QDialog`, `QLabel`, etc.).
- Accept plain Python values; return plain Python values; raise exceptions on failure.
- Controllers do not open dialogs. They do not hold references to views.
- Communicate outward only via signals (`result_ready`, `progress`, `batch_done`, etc.).

**`views/` — Widgets and presentation**
- Composed of `QWidget` subclasses. Display model data; never store canonical application state.
- `QFileDialog`, `QMessageBox`, `QColorDialog`, `QInputDialog` live here — Qt dialogs require a parent widget and their return values are view-level input, not business logic.
- Sub-widgets (sections, rows, panels) define their contract through **public methods and signals only**. Parent widgets must never access `_private` attributes or internal lists of a child widget. If a parent needs to trigger internal logic, the child must expose a named public method.

---

## Communication

| Direction | Rule |
|-----------|------|
| Model → View | Model signals only (`dataChanged`, `modelReset`, custom signals). Models never hold view references. |
| View → Model (read) | Call model query methods directly (`get_annotations(row)`, `get_class_names()`, etc.). |
| View → Model (write) | Leaf-level atomic mutations (`delete_annotation`, `set_class_color`) may be called directly from sub-widgets. Compound or multi-step operations must go through the controller. |
| View → Controller | Call controller methods directly, or emit a signal that the top-level window routes to the controller. Leaf sub-widgets do not import controllers. |
| View ↔ View | Sibling views never call each other directly. Coordinate through shared model signals or parent-level signal routing. |

---

## Type Conversions

- Domain layer (`core/`, `models/`): plain Python types only — `tuple`, `str`, `int`, `float`, `list`.
- Qt types (`QColor`, `QPixmap`, `QPen`, `QRect`) are constructed **immediately before use** at the widget paint boundary. They are never stored in models or states.

---

## Signals and Slots

- Signals express **user intent** ("the user requested delete") not implementation steps ("call `_rebuild`").
- One signal = one logical event. Do not multiplex different events through a single signal distinguished by a flag.
- Guard reciprocal cross-view sync with a `_syncing: bool` flag to prevent feedback loops.
- Debounce high-frequency signals (sliders, live text input) with a single-shot `QTimer` before triggering expensive recomputes.
