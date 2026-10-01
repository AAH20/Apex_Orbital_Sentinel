"""Space C2 — Command and Control, Tasking, and Coordination.

Provides a modular C2 system for space operations including:
- Operator registration and federation
- Command lifecycle management (issue, acknowledge, complete, fail, cancel)
- Task management with priorities, dependencies, and assignment
- Resource allocation and capacity tracking
- Multi-operator coordination and conflict detection
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class CommandStatus(Enum):
    """Status of a C2 command."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Status of a C2 task."""
    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class Command:
    """A C2 command issued from one operator to another."""

    def __init__(
        self,
        issuer: str,
        recipient: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.issuer = issuer
        self.recipient = recipient
        self.action = action
        self.params = params or {}
        self.status = CommandStatus.PENDING
        self.acknowledged_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.failure_reason: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return (
            f"Command(id={self.id[:8]}..., {self.issuer}->{self.recipient}, "
            f"action={self.action}, status={self.status.value})"
        )


class Task:
    """A C2 task with priority, assignment, and dependency tracking."""

    def __init__(
        self,
        name: str,
        description: str,
        priority: TaskPriority,
        assignee: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.priority = priority
        self.assignee = assignee
        self.dependencies = dependencies or []
        self.status = TaskStatus.CREATED
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.block_reason: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id[:8]}..., name={self.name}, "
            f"priority={self.priority.name}, status={self.status.value})"
        )


class Resource:
    """A trackable resource with capacity management."""

    def __init__(self, resource_id: str, name: str, resource_type: str, capacity: float):
        self.id = resource_id
        self.name = name
        self.type = resource_type
        self.capacity = capacity
        self.allocated = 0.0

    @property
    def available(self) -> float:
        return self.capacity - self.allocated

    def __repr__(self) -> str:
        return (
            f"Resource(id={self.id}, name={self.name}, "
            f"allocated={self.allocated}/{self.capacity})"
        )


class C2System:
    """Space Command and Control system.

    Manages operators, commands, tasks, and resources for coordinated
    space operations.
    """

    def __init__(self):
        self.operators: Dict[str, Dict[str, Any]] = {}
        self.commands: Dict[str, Command] = {}
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        self._command_tasks: Dict[str, List[str]] = {}  # command_id -> [task_ids]

    # ── Operator Management ──────────────────────────────────────────

    def register_operator(self, name: str) -> bool:
        """Register a new operator. Returns False if already registered."""
        if name in self.operators:
            return False
        self.operators[name] = {
            "registered_at": datetime.now(timezone.utc),
            "active_commands": [],
            "active_tasks": [],
        }
        return True

    # ── Command Lifecycle ────────────────────────────────────────────

    def issue_command(
        self,
        issuer: str,
        recipient: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Command:
        """Issue a command from one operator to another."""
        if issuer not in self.operators:
            raise ValueError(f"Unregistered issuer: {issuer}")
        if recipient not in self.operators:
            raise ValueError(f"Unregistered recipient: {recipient}")

        cmd = Command(issuer, recipient, action, params)
        self.commands[cmd.id] = cmd
        self.operators[issuer]["active_commands"].append(cmd.id)
        self.operators[recipient]["active_commands"].append(cmd.id)
        return cmd

    def acknowledge_command(self, command_id: str) -> bool:
        """Acknowledge a pending command."""
        cmd = self.commands.get(command_id)
        if cmd is None or cmd.status != CommandStatus.PENDING:
            return False
        cmd.status = CommandStatus.ACKNOWLEDGED
        cmd.acknowledged_at = datetime.now(timezone.utc)
        return True

    def complete_command(self, command_id: str) -> bool:
        """Complete an acknowledged command."""
        cmd = self.commands.get(command_id)
        if cmd is None or cmd.status != CommandStatus.ACKNOWLEDGED:
            return False
        cmd.status = CommandStatus.COMPLETED
        cmd.completed_at = datetime.now(timezone.utc)
        return True

    def fail_command(self, command_id: str, reason: str) -> bool:
        """Fail an acknowledged command with a reason."""
        cmd = self.commands.get(command_id)
        if cmd is None or cmd.status != CommandStatus.ACKNOWLEDGED:
            return False
        cmd.status = CommandStatus.FAILED
        cmd.failure_reason = reason
        cmd.completed_at = datetime.now(timezone.utc)
        return True

    def cancel_command(self, command_id: str) -> bool:
        """Cancel a pending or acknowledged command."""
        cmd = self.commands.get(command_id)
        if cmd is None:
            return False
        if cmd.status in (CommandStatus.COMPLETED, CommandStatus.FAILED, CommandStatus.CANCELLED):
            return False
        cmd.status = CommandStatus.CANCELLED
        return True

    def get_commands_by_recipient(self, recipient: str) -> List[Command]:
        """Get all commands addressed to a specific operator."""
        return [cmd for cmd in self.commands.values() if cmd.recipient == recipient]

    def get_commands_by_status(self, status: CommandStatus) -> List[Command]:
        """Get all commands with a specific status."""
        return [cmd for cmd in self.commands.values() if cmd.status == status]

    # ── Task Management ──────────────────────────────────────────────

    def create_task(
        self,
        name: str,
        description: str,
        priority: TaskPriority,
        assignee: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Task:
        """Create a new task."""
        deps = dependencies or []
        for dep_id in deps:
            if dep_id not in self.tasks:
                raise ValueError(f"Dependency task not found: {dep_id}")

        task = Task(name, description, priority, assignee, deps)
        self.tasks[task.id] = task
        if assignee and assignee in self.operators:
            self.operators[assignee]["active_tasks"].append(task.id)
        return task

    def assign_task(self, task_id: str, assignee: str) -> bool:
        """Assign a task to an operator."""
        task = self.tasks.get(task_id)
        if task is None:
            return False
        task.assignee = assignee
        task.status = TaskStatus.ASSIGNED
        if assignee in self.operators:
            self.operators[assignee]["active_tasks"].append(task_id)
        return True

    def start_task(self, task_id: str) -> bool:
        """Start an assigned task."""
        task = self.tasks.get(task_id)
        if task is None or task.assignee is None:
            return False
        if task.status not in (TaskStatus.ASSIGNED, TaskStatus.CREATED):
            return False
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now(timezone.utc)
        return True

    def complete_task(self, task_id: str) -> bool:
        """Complete an in-progress task."""
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.IN_PROGRESS:
            return False
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        return True

    def block_task(self, task_id: str, reason: str) -> bool:
        """Block an in-progress task."""
        task = self.tasks.get(task_id)
        if task is None or task.status != TaskStatus.IN_PROGRESS:
            return False
        task.status = TaskStatus.BLOCKED
        task.block_reason = reason
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task that is not yet completed."""
        task = self.tasks.get(task_id)
        if task is None:
            return False
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED):
            return False
        task.status = TaskStatus.CANCELLED
        return True

    def get_tasks_by_priority(self, priority: TaskPriority) -> List[Task]:
        """Get all tasks with a specific priority."""
        return [t for t in self.tasks.values() if t.priority == priority]

    def get_tasks_by_assignee(self, assignee: str) -> List[Task]:
        """Get all tasks assigned to a specific operator."""
        return [t for t in self.tasks.values() if t.assignee == assignee]

    # ── Resource Management ──────────────────────────────────────────

    def add_resource(
        self, resource_id: str, name: str, resource_type: str, capacity: float
    ) -> Optional[Resource]:
        """Add a new resource. Returns None if ID already exists."""
        if resource_id in self.resources:
            return None
        res = Resource(resource_id, name, resource_type, capacity)
        self.resources[resource_id] = res
        return res

    def allocate_resource(self, resource_id: str, amount: float) -> bool:
        """Allocate capacity from a resource."""
        res = self.resources.get(resource_id)
        if res is None or amount < 0 or res.available < amount:
            return False
        res.allocated += amount
        return True

    def deallocate_resource(self, resource_id: str, amount: float) -> bool:
        """Deallocate capacity from a resource."""
        res = self.resources.get(resource_id)
        if res is None or amount < 0 or res.allocated < amount:
            return False
        res.allocated -= amount
        return True

    def get_available_capacity(self, resource_id: str) -> Optional[float]:
        """Get available capacity for a resource."""
        res = self.resources.get(resource_id)
        if res is None:
            return None
        return res.available

    # ── Coordination ──────────────────────────────────────────────────

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect scheduling conflicts (same assignee, overlapping high-priority tasks)."""
        conflicts = []
        assignee_tasks: Dict[str, List[Task]] = {}
        for task in self.tasks.values():
            if task.assignee and task.status not in (
                TaskStatus.COMPLETED,
                TaskStatus.CANCELLED,
            ):
                assignee_tasks.setdefault(task.assignee, []).append(task)

        for assignee, tasks in assignee_tasks.items():
            if len(tasks) > 1:
                high_priority = [t for t in tasks if t.priority in (TaskPriority.HIGH, TaskPriority.CRITICAL)]
                if len(high_priority) > 1:
                    conflicts.append({
                        "type": "resource_contention",
                        "assignee": assignee,
                        "tasks": [t.id for t in high_priority],
                        "description": f"Operator {assignee} has {len(high_priority)} high-priority active tasks",
                    })
        return conflicts

    def coordinate_tasking(self, task_ids: List[str]) -> Dict[str, Any]:
        """Coordinate execution of multiple tasks."""
        if not task_ids:
            return {"status": "no_tasks", "tasks": []}

        tasks = []
        blocked = []
        for tid in task_ids:
            task = self.tasks.get(tid)
            if task is None:
                return {"status": "error", "message": f"Task not found: {tid}"}
            tasks.append(task)
            if task.status == TaskStatus.BLOCKED:
                blocked.append(tid)

        return {
            "status": "coordinated",
            "tasks": [t.id for t in tasks],
            "blocked": blocked,
            "total": len(tasks),
        }

    # ── Command-Task Integration ─────────────────────────────────────

    def link_command_to_task(self, command_id: str, task_id: str) -> bool:
        """Link a command to a task."""
        if command_id not in self.commands or task_id not in self.tasks:
            return False
        self._command_tasks.setdefault(command_id, []).append(task_id)
        return True

    def get_tasks_for_command(self, command_id: str) -> List[Task]:
        """Get all tasks linked to a command."""
        task_ids = self._command_tasks.get(command_id, [])
        return [self.tasks[tid] for tid in task_ids if tid in self.tasks]
