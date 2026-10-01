"""Unit tests for Space C2 — command and control, tasking, coordination."""

import unittest
from datetime import datetime, timezone

from src.space.c2 import (
    C2System,
    Command,
    CommandStatus,
    Resource,
    Task,
    TaskPriority,
    TaskStatus,
)


class TestOperatorRegistration(unittest.TestCase):
    """C2 operator registration and federation."""

    def setUp(self):
        self.c2 = C2System()

    def test_register_operator(self):
        result = self.c2.register_operator("OP-Alpha")
        self.assertTrue(result)
        self.assertIn("OP-Alpha", self.c2.operators)

    def test_register_duplicate_operator(self):
        self.c2.register_operator("OP-Alpha")
        result = self.c2.register_operator("OP-Alpha")
        self.assertFalse(result)
        self.assertEqual(len(self.c2.operators), 1)

    def test_register_multiple_operators(self):
        self.c2.register_operator("OP-Alpha")
        self.c2.register_operator("OP-Bravo")
        self.assertEqual(len(self.c2.operators), 2)


class TestCommandLifecycle(unittest.TestCase):
    """Command issuance, acknowledgment, completion, failure, cancellation."""

    def setUp(self):
        self.c2 = C2System()
        self.c2.register_operator("OP-Alpha")
        self.c2.register_operator("OP-Bravo")

    def test_issue_command(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {"target_id": "SAT-1"})
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.issuer, "OP-Alpha")
        self.assertEqual(cmd.recipient, "OP-Bravo")
        self.assertEqual(cmd.action, "TRACK")
        self.assertEqual(cmd.status, CommandStatus.PENDING)
        self.assertIn(cmd.id, self.c2.commands)

    def test_issue_command_unregistered_issuer(self):
        with self.assertRaises(ValueError):
            self.c2.issue_command("OP-Unknown", "OP-Bravo", "TRACK", {})

    def test_issue_command_unregistered_recipient(self):
        with self.assertRaises(ValueError):
            self.c2.issue_command("OP-Alpha", "OP-Unknown", "TRACK", {})

    def test_acknowledge_command(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        result = self.c2.acknowledge_command(cmd.id)
        self.assertTrue(result)
        self.assertEqual(cmd.status, CommandStatus.ACKNOWLEDGED)
        self.assertIsNotNone(cmd.acknowledged_at)

    def test_acknowledge_nonexistent_command(self):
        result = self.c2.acknowledge_command("nonexistent-id")
        self.assertFalse(result)

    def test_complete_command(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        self.c2.acknowledge_command(cmd.id)
        result = self.c2.complete_command(cmd.id)
        self.assertTrue(result)
        self.assertEqual(cmd.status, CommandStatus.COMPLETED)
        self.assertIsNotNone(cmd.completed_at)

    def test_complete_without_acknowledge(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        result = self.c2.complete_command(cmd.id)
        self.assertFalse(result)
        self.assertEqual(cmd.status, CommandStatus.PENDING)

    def test_fail_command(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        self.c2.acknowledge_command(cmd.id)
        result = self.c2.fail_command(cmd.id, "Sensor offline")
        self.assertTrue(result)
        self.assertEqual(cmd.status, CommandStatus.FAILED)

    def test_cancel_command(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        result = self.c2.cancel_command(cmd.id)
        self.assertTrue(result)
        self.assertEqual(cmd.status, CommandStatus.CANCELLED)

    def test_cancel_completed_command(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        self.c2.acknowledge_command(cmd.id)
        self.c2.complete_command(cmd.id)
        result = self.c2.cancel_command(cmd.id)
        self.assertFalse(result)
        self.assertEqual(cmd.status, CommandStatus.COMPLETED)

    def test_get_commands_by_recipient(self):
        cmd1 = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        cmd2 = self.c2.issue_command("OP-Alpha", "OP-Bravo", "IMAGE", {})
        cmd3 = self.c2.issue_command("OP-Bravo", "OP-Alpha", "TRACK", {})
        bravo_cmds = self.c2.get_commands_by_recipient("OP-Bravo")
        self.assertEqual(len(bravo_cmds), 2)
        self.assertIn(cmd1, bravo_cmds)
        self.assertIn(cmd2, bravo_cmds)

    def test_get_commands_by_status(self):
        cmd1 = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        cmd2 = self.c2.issue_command("OP-Alpha", "OP-Bravo", "IMAGE", {})
        self.c2.acknowledge_command(cmd1.id)
        self.c2.complete_command(cmd1.id)
        completed = self.c2.get_commands_by_status(CommandStatus.COMPLETED)
        self.assertEqual(len(completed), 1)
        self.assertIn(cmd1, completed)


class TestTaskManagement(unittest.TestCase):
    """Task creation, assignment, execution, and lifecycle."""

    def setUp(self):
        self.c2 = C2System()
        self.c2.register_operator("OP-Alpha")
        self.c2.register_operator("OP-Bravo")

    def test_create_task(self):
        task = self.c2.create_task("Track debris", "Track object DEB-4421", TaskPriority.HIGH)
        self.assertIsNotNone(task)
        self.assertEqual(task.name, "Track debris")
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.status, TaskStatus.CREATED)
        self.assertIn(task.id, self.c2.tasks)

    def test_create_task_with_assignee(self):
        task = self.c2.create_task("Image region", "Image region R-7", TaskPriority.MEDIUM, assignee="OP-Bravo")
        self.assertEqual(task.assignee, "OP-Bravo")

    def test_create_task_with_dependencies(self):
        dep_task = self.c2.create_task("Pre-task", "Setup", TaskPriority.LOW)
        task = self.c2.create_task("Main task", "Execute", TaskPriority.HIGH, dependencies=[dep_task.id])
        self.assertIn(dep_task.id, task.dependencies)

    def test_create_task_invalid_dependency(self):
        with self.assertRaises(ValueError):
            self.c2.create_task("Bad task", "Execute", TaskPriority.HIGH, dependencies=["nonexistent"])

    def test_assign_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH)
        result = self.c2.assign_task(task.id, "OP-Bravo")
        self.assertTrue(result)
        self.assertEqual(task.assignee, "OP-Bravo")
        self.assertEqual(task.status, TaskStatus.ASSIGNED)

    def test_assign_nonexistent_task(self):
        result = self.c2.assign_task("nonexistent", "OP-Bravo")
        self.assertFalse(result)

    def test_start_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH, assignee="OP-Bravo")
        result = self.c2.start_task(task.id)
        self.assertTrue(result)
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
        self.assertIsNotNone(task.started_at)

    def test_start_unassigned_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH)
        result = self.c2.start_task(task.id)
        self.assertFalse(result)
        self.assertEqual(task.status, TaskStatus.CREATED)

    def test_complete_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH, assignee="OP-Bravo")
        self.c2.start_task(task.id)
        result = self.c2.complete_task(task.id)
        self.assertTrue(result)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(task.completed_at)

    def test_complete_blocked_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH, assignee="OP-Bravo")
        self.c2.start_task(task.id)
        self.c2.block_task(task.id, "Sensor unavailable")
        result = self.c2.complete_task(task.id)
        self.assertFalse(result)
        self.assertEqual(task.status, TaskStatus.BLOCKED)

    def test_block_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH, assignee="OP-Bravo")
        self.c2.start_task(task.id)
        result = self.c2.block_task(task.id, "Fuel low")
        self.assertTrue(result)
        self.assertEqual(task.status, TaskStatus.BLOCKED)

    def test_cancel_task(self):
        task = self.c2.create_task("Track", "Track SAT-1", TaskPriority.HIGH)
        result = self.c2.cancel_task(task.id)
        self.assertTrue(result)
        self.assertEqual(task.status, TaskStatus.CANCELLED)

    def test_cancel_nonexistent_task(self):
        result = self.c2.cancel_task("nonexistent")
        self.assertFalse(result)

    def test_get_tasks_by_priority(self):
        task_low = self.c2.create_task("Low", "Low priority", TaskPriority.LOW)
        task_high = self.c2.create_task("High", "High priority", TaskPriority.HIGH)
        task_crit = self.c2.create_task("Critical", "Critical priority", TaskPriority.CRITICAL)
        high_tasks = self.c2.get_tasks_by_priority(TaskPriority.HIGH)
        self.assertEqual(len(high_tasks), 1)
        self.assertIn(task_high, high_tasks)

    def test_get_tasks_by_assignee(self):
        task1 = self.c2.create_task("T1", "Task 1", TaskPriority.HIGH, assignee="OP-Bravo")
        task2 = self.c2.create_task("T2", "Task 2", TaskPriority.MEDIUM, assignee="OP-Bravo")
        task3 = self.c2.create_task("T3", "Task 3", TaskPriority.LOW, assignee="OP-Alpha")
        bravo_tasks = self.c2.get_tasks_by_assignee("OP-Bravo")
        self.assertEqual(len(bravo_tasks), 2)


class TestResourceManagement(unittest.TestCase):
    """Resource allocation and capacity management."""

    def setUp(self):
        self.c2 = C2System()

    def test_add_resource(self):
        res = self.c2.add_resource("RES-1", "Power", "power", 100.0)
        self.assertIsNotNone(res)
        self.assertEqual(res.name, "Power")
        self.assertEqual(res.capacity, 100.0)
        self.assertEqual(res.allocated, 0.0)
        self.assertIn(res.id, self.c2.resources)

    def test_add_duplicate_resource(self):
        self.c2.add_resource("RES-1", "Power", "power", 100.0)
        result = self.c2.add_resource("RES-1", "Power", "power", 100.0)
        self.assertFalse(result)

    def test_allocate_resource(self):
        self.c2.add_resource("RES-1", "Power", "power", 100.0)
        result = self.c2.allocate_resource("RES-1", 30.0)
        self.assertTrue(result)
        self.assertEqual(self.c2.resources["RES-1"].allocated, 30.0)

    def test_allocate_resource_over_capacity(self):
        self.c2.add_resource("RES-1", "Power", "power", 100.0)
        result = self.c2.allocate_resource("RES-1", 150.0)
        self.assertFalse(result)
        self.assertEqual(self.c2.resources["RES-1"].allocated, 0.0)

    def test_allocate_nonexistent_resource(self):
        result = self.c2.allocate_resource("nonexistent", 10.0)
        self.assertFalse(result)

    def test_deallocate_resource(self):
        self.c2.add_resource("RES-1", "Power", "power", 100.0)
        self.c2.allocate_resource("RES-1", 50.0)
        result = self.c2.deallocate_resource("RES-1", 20.0)
        self.assertTrue(result)
        self.assertEqual(self.c2.resources["RES-1"].allocated, 30.0)

    def test_deallocate_more_than_allocated(self):
        self.c2.add_resource("RES-1", "Power", "power", 100.0)
        self.c2.allocate_resource("RES-1", 30.0)
        result = self.c2.deallocate_resource("RES-1", 50.0)
        self.assertFalse(result)
        self.assertEqual(self.c2.resources["RES-1"].allocated, 30.0)

    def test_get_available_capacity(self):
        self.c2.add_resource("RES-1", "Power", "power", 100.0)
        self.c2.allocate_resource("RES-1", 40.0)
        available = self.c2.get_available_capacity("RES-1")
        self.assertEqual(available, 60.0)

    def test_get_available_capacity_nonexistent(self):
        result = self.c2.get_available_capacity("nonexistent")
        self.assertIsNone(result)


class TestCoordination(unittest.TestCase):
    """Multi-operator coordination and conflict detection."""

    def setUp(self):
        self.c2 = C2System()
        self.c2.register_operator("OP-Alpha")
        self.c2.register_operator("OP-Bravo")

    def test_detect_conflicts_no_conflicts(self):
        self.c2.create_task("T1", "Task 1", TaskPriority.HIGH, assignee="OP-Alpha")
        self.c2.create_task("T2", "Task 2", TaskPriority.HIGH, assignee="OP-Bravo")
        conflicts = self.c2.detect_conflicts()
        self.assertEqual(len(conflicts), 0)

    def test_detect_conflicts_same_assignee(self):
        self.c2.create_task("T1", "Task 1", TaskPriority.HIGH, assignee="OP-Alpha")
        self.c2.create_task("T2", "Task 2", TaskPriority.HIGH, assignee="OP-Alpha")
        conflicts = self.c2.detect_conflicts()
        self.assertGreater(len(conflicts), 0)

    def test_coordinate_tasking(self):
        task1 = self.c2.create_task("T1", "Task 1", TaskPriority.HIGH, assignee="OP-Alpha")
        task2 = self.c2.create_task("T2", "Task 2", TaskPriority.MEDIUM, assignee="OP-Bravo")
        result = self.c2.coordinate_tasking([task1.id, task2.id])
        self.assertIn("status", result)
        self.assertIn("tasks", result)

    def test_coordinate_tasking_with_blocked(self):
        task1 = self.c2.create_task("T1", "Task 1", TaskPriority.HIGH, assignee="OP-Alpha")
        task2 = self.c2.create_task("T2", "Task 2", TaskPriority.HIGH, assignee="OP-Bravo")
        self.c2.start_task(task1.id)
        self.c2.block_task(task1.id, "Resource unavailable")
        result = self.c2.coordinate_tasking([task1.id, task2.id])
        self.assertIn("blocked", result)

    def test_coordinate_tasking_empty(self):
        result = self.c2.coordinate_tasking([])
        self.assertEqual(result["status"], "no_tasks")

    def test_coordinate_tasking_nonexistent(self):
        result = self.c2.coordinate_tasking(["nonexistent"])
        self.assertEqual(result["status"], "error")


class TestCommandTaskIntegration(unittest.TestCase):
    """Integration between commands and tasks."""

    def setUp(self):
        self.c2 = C2System()
        self.c2.register_operator("OP-Alpha")
        self.c2.register_operator("OP-Bravo")

    def test_command_creates_task(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {"target_id": "SAT-1"})
        task = self.c2.create_task("Track SAT-1", "Track satellite SAT-1", TaskPriority.HIGH, assignee="OP-Bravo")
        self.c2.link_command_to_task(cmd.id, task.id)
        linked_tasks = self.c2.get_tasks_for_command(cmd.id)
        self.assertIn(task, linked_tasks)

    def test_link_command_to_nonexistent_task(self):
        cmd = self.c2.issue_command("OP-Alpha", "OP-Bravo", "TRACK", {})
        result = self.c2.link_command_to_task(cmd.id, "nonexistent")
        self.assertFalse(result)

    def test_get_tasks_for_nonexistent_command(self):
        result = self.c2.get_tasks_for_command("nonexistent")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
