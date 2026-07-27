// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

/// @title AgentRegistry
/// @notice Binds each simulated LLM agent to a verifiable on-chain identity
/// and stores its reputation, capability vector, and workload.
/// Matches methodology Steps 2 (registration) and 19 (state sync), and
/// paper Section IV-A (Registration and Authentication Module).
///
/// All float-valued quantities from the paper (reputation in [0,1],
/// capability weights in [0,1], performance scores in [0,1]) are stored as
/// uint256 scaled by SCALE=10000, since Solidity has no floating point.
contract AgentRegistry {
    uint256 public constant NUM_CAPABILITY_TAGS = 10;
    uint256 public constant SCALE = 10000;

    struct Agent {
        uint256 agentId;
        address owner; // the agent's own secp256k1 key pair
        uint256 reputation; // scaled by SCALE
        uint256[NUM_CAPABILITY_TAGS] capabilities; // each scaled by SCALE
        uint256 workload;
        bool exists;
    }

    address public admin;
    mapping(uint256 => Agent) private agents;
    mapping(address => uint256) public ownerToAgentId; // 0 = unbound
    mapping(address => bool) public authorizedCallers; // TaskManager / IncentiveContract

    event AgentRegistered(uint256 indexed agentId, address indexed owner, uint256 initialReputation);
    event ReputationUpdated(uint256 indexed agentId, uint256 oldReputation, uint256 newReputation);
    event CapabilityUpdated(uint256 indexed agentId, uint256 indexed tagIndex, uint256 oldValue, uint256 newValue);
    event WorkloadUpdated(uint256 indexed agentId, uint256 oldWorkload, uint256 newWorkload);
    event CallerAuthorized(address indexed caller, bool authorized);

    modifier onlyAdmin() {
        require(msg.sender == admin, "AgentRegistry: not admin");
        _;
    }

    modifier onlyAuthorized() {
        require(authorizedCallers[msg.sender], "AgentRegistry: caller not authorized");
        _;
    }

    constructor() {
        admin = msg.sender;
    }

    /// @notice Admin authorizes TaskManager / IncentiveContract to mutate
    /// state on agents' behalf (agents themselves cannot self-report
    /// reputation or capability, only register their own identity).
    function setAuthorizedCaller(address caller, bool authorized) external onlyAdmin {
        authorizedCallers[caller] = authorized;
        emit CallerAuthorized(caller, authorized);
    }

    /// @notice Step 2: each agent registers itself using its own key pair.
    function registerAgent(
        uint256 agentId,
        uint256[NUM_CAPABILITY_TAGS] calldata initialCapabilities,
        uint256 initialReputation
    ) external {
        require(!agents[agentId].exists, "AgentRegistry: agent already registered");
        require(ownerToAgentId[msg.sender] == 0, "AgentRegistry: key already bound to an agent");
        require(initialReputation <= SCALE, "AgentRegistry: reputation > 1.0");

        Agent storage a = agents[agentId];
        a.agentId = agentId;
        a.owner = msg.sender;
        a.reputation = initialReputation;
        a.capabilities = initialCapabilities;
        a.workload = 0;
        a.exists = true;

        ownerToAgentId[msg.sender] = agentId;

        emit AgentRegistered(agentId, msg.sender, initialReputation);
    }

    function updateReputation(uint256 agentId, uint256 newReputation) external onlyAuthorized {
        Agent storage a = agents[agentId];
        require(a.exists, "AgentRegistry: unknown agent");
        require(newReputation <= SCALE, "AgentRegistry: reputation > 1.0");
        uint256 old = a.reputation;
        a.reputation = newReputation;
        emit ReputationUpdated(agentId, old, newReputation);
    }

    function updateCapability(uint256 agentId, uint256 tagIndex, uint256 newValue) external onlyAuthorized {
        require(tagIndex < NUM_CAPABILITY_TAGS, "AgentRegistry: bad tag index");
        Agent storage a = agents[agentId];
        require(a.exists, "AgentRegistry: unknown agent");
        require(newValue <= SCALE, "AgentRegistry: capability > 1.0");
        uint256 old = a.capabilities[tagIndex];
        a.capabilities[tagIndex] = newValue;
        emit CapabilityUpdated(agentId, tagIndex, old, newValue);
    }

    function setWorkload(uint256 agentId, uint256 newWorkload) external onlyAuthorized {
        Agent storage a = agents[agentId];
        require(a.exists, "AgentRegistry: unknown agent");
        uint256 old = a.workload;
        a.workload = newWorkload;
        emit WorkloadUpdated(agentId, old, newWorkload);
    }

    function getReputation(uint256 agentId) external view returns (uint256) {
        require(agents[agentId].exists, "AgentRegistry: unknown agent");
        return agents[agentId].reputation;
    }

    function getCapability(uint256 agentId, uint256 tagIndex) external view returns (uint256) {
        require(agents[agentId].exists, "AgentRegistry: unknown agent");
        require(tagIndex < NUM_CAPABILITY_TAGS, "AgentRegistry: bad tag index");
        return agents[agentId].capabilities[tagIndex];
    }

    function getCapabilities(uint256 agentId) external view returns (uint256[NUM_CAPABILITY_TAGS] memory) {
        require(agents[agentId].exists, "AgentRegistry: unknown agent");
        return agents[agentId].capabilities;
    }

    function getWorkload(uint256 agentId) external view returns (uint256) {
        require(agents[agentId].exists, "AgentRegistry: unknown agent");
        return agents[agentId].workload;
    }

    function agentExists(uint256 agentId) external view returns (bool) {
        return agents[agentId].exists;
    }
}
