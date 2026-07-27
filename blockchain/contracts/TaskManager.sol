// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

interface IAgentRegistryTaskManager {
    function setWorkload(uint256 agentId, uint256 newWorkload) external;
    function getWorkload(uint256 agentId) external view returns (uint256);
    function agentExists(uint256 agentId) external view returns (bool);
}

/// @title TaskManager
/// @notice Records task metadata (Step 4), assignment (Step 10), behavioral
/// logs, and completion (Step 16), and reconciles workload (Step 18).
/// Matches paper Section IV-C (Task Allocation Module) + Execution Log
/// contract shown in Fig. 2.
///
/// In this simulation, the "coordinator" (the off-chain Python relay that
/// runs the auction) is the only caller of these functions - it publishes
/// the outcome of off-chain computation (auction, evaluation) as verifiable
/// on-chain state, matching the paper's hybrid on-chain/off-chain design
/// (Section IV: "computation-intensive operations ... remain off-chain").
contract TaskManager {
    struct TaskMetadata {
        uint256 taskId;
        uint256 reward; // scaled by REWARD_SCALE (100), i.e. 2 decimal places
        bytes32 taskHash; // keccak256 of the canonical task payload
        uint256 deadline;
        uint256 assignedAgent;
        bool assigned;
        bool completed;
        bool success;
        bool exists;
    }

    uint256 public constant REWARD_SCALE = 100;

    address public admin;
    IAgentRegistryTaskManager public registry;

    mapping(uint256 => TaskMetadata) public tasks;

    event TaskSubmitted(uint256 indexed taskId, uint256 reward, bytes32 taskHash, uint256 deadline);
    event TaskAssigned(uint256 indexed taskId, uint256 indexed agentId, uint256 scoreScaled);
    event ActionLogged(uint256 indexed agentId, uint256 indexed taskId, string action, uint256 timestamp);
    event TaskCompleted(uint256 indexed taskId, uint256 indexed agentId, bool success, uint256 qualityScaled);

    modifier onlyAdmin() {
        require(msg.sender == admin, "TaskManager: not admin");
        _;
    }

    constructor(address registryAddress) {
        admin = msg.sender;
        registry = IAgentRegistryTaskManager(registryAddress);
    }

    /// @notice Step 4: register task metadata on-chain, including a hash of
    /// the full task payload for tamper detection.
    function submitTask(uint256 taskId, uint256 reward, bytes32 taskHash, uint256 deadline) external onlyAdmin {
        require(!tasks[taskId].exists, "TaskManager: task already submitted");
        tasks[taskId] = TaskMetadata({
            taskId: taskId,
            reward: reward,
            taskHash: taskHash,
            deadline: deadline,
            assignedAgent: 0,
            assigned: false,
            completed: false,
            success: false,
            exists: true
        });
        emit TaskSubmitted(taskId, reward, taskHash, deadline);
    }

    /// @notice Step 10: commit the auction winner and bump its workload.
    function assignTask(uint256 taskId, uint256 agentId, uint256 scoreScaled) external onlyAdmin {
        TaskMetadata storage t = tasks[taskId];
        require(t.exists, "TaskManager: unknown task");
        require(!t.assigned, "TaskManager: already assigned");
        require(registry.agentExists(agentId), "TaskManager: unknown agent");

        t.assignedAgent = agentId;
        t.assigned = true;

        uint256 currentWorkload = registry.getWorkload(agentId);
        registry.setWorkload(agentId, currentWorkload + 1);

        emit TaskAssigned(taskId, agentId, scoreScaled);
    }

    /// @notice Behavioral evidence log: binds <agent, task, action, timestamp>
    /// tuples as an audit trail (paper Section IV: logAction interface).
    function logAction(uint256 agentId, uint256 taskId, string calldata action) external onlyAdmin {
        emit ActionLogged(agentId, taskId, action, block.timestamp);
    }

    /// @notice Step 16: mark task success/failure. Workload is NOT touched
    /// here; the coordinator reconciles workload once per round via
    /// syncWorkload (Step 18), matching the simulation's round-based
    /// release-one-unit-per-agent semantics.
    function completeTask(uint256 taskId, bool success, uint256 qualityScaled) external onlyAdmin {
        TaskMetadata storage t = tasks[taskId];
        require(t.exists, "TaskManager: unknown task");
        require(t.assigned, "TaskManager: not assigned");
        require(!t.completed, "TaskManager: already completed");

        t.completed = true;
        t.success = success;

        emit TaskCompleted(taskId, t.assignedAgent, success, qualityScaled);
    }

    /// @notice Step 18: reconcile an agent's workload counter on-chain
    /// (called once per round for every agent by the coordinator).
    function syncWorkload(uint256 agentId, uint256 newWorkload) external onlyAdmin {
        require(registry.agentExists(agentId), "TaskManager: unknown agent");
        registry.setWorkload(agentId, newWorkload);
    }

    function getTask(uint256 taskId) external view returns (TaskMetadata memory) {
        return tasks[taskId];
    }
}
