// SPDX-License-Identifier: MIT
pragma solidity 0.8.20;

interface IAgentRegistryIncentive {
    function getReputation(uint256 agentId) external view returns (uint256);
    function getCapability(uint256 agentId, uint256 tagIndex) external view returns (uint256);
    function updateReputation(uint256 agentId, uint256 newReputation) external;
    function updateCapability(uint256 agentId, uint256 tagIndex, uint256 newValue) external;
    function agentExists(uint256 agentId) external view returns (bool);
}

/// @title IncentiveContract
/// @notice Enforces the reputation EMA (Eq. 4, methodology Step 14) and
/// capability-weight EMA (Eq. 7, methodology Step 15) ON-CHAIN, given an
/// off-chain-computed performance score S_i^t (Eq. 5 / Step 13), scaled by
/// SCALE=10000. This is the "incentive-relevant state transition" the paper
/// (Section IV-D) requires to be irrevocable and transparent - the update
/// arithmetic itself, not just its result, is committed on-chain.
contract IncentiveContract {
    uint256 public constant SCALE = 10000;

    address public admin;
    IAgentRegistryIncentive public registry;

    uint256 public lambdaRep = 7000; // reputation smoothing factor (lambda), 0.7
    uint256 public muCap = 6000; // capability smoothing factor (mu), 0.6

    event ReputationUpdated(uint256 indexed agentId, uint256 oldReputation, uint256 newReputation, uint256 taskScore);
    event CapabilityUpdated(uint256 indexed agentId, uint256 indexed tagIndex, uint256 oldValue, uint256 newValue, uint256 taskScore);

    modifier onlyAdmin() {
        require(msg.sender == admin, "IncentiveContract: not admin");
        _;
    }

    constructor(address registryAddress) {
        admin = msg.sender;
        registry = IAgentRegistryIncentive(registryAddress);
    }

    function setSmoothingFactors(uint256 newLambdaRep, uint256 newMuCap) external onlyAdmin {
        require(newLambdaRep <= SCALE && newMuCap <= SCALE, "IncentiveContract: factor > 1.0");
        lambdaRep = newLambdaRep;
        muCap = newMuCap;
    }

    /// @notice rho_i^{t+1} = lambda * rho_i^t + (1 - lambda) * S_i^t   (Eq. 4)
    function updateReputation(uint256 agentId, uint256 taskScoreScaled) external onlyAdmin returns (uint256) {
        require(registry.agentExists(agentId), "IncentiveContract: unknown agent");
        require(taskScoreScaled <= SCALE, "IncentiveContract: score > 1.0");

        uint256 oldRep = registry.getReputation(agentId);
        uint256 newRep = (lambdaRep * oldRep + (SCALE - lambdaRep) * taskScoreScaled) / SCALE;

        registry.updateReputation(agentId, newRep);
        emit ReputationUpdated(agentId, oldRep, newRep, taskScoreScaled);
        return newRep;
    }

    /// @notice w_ik^{t+1} = mu * w_ik^t + (1 - mu) * S_i^t   (Eq. 7)
    function updateCapability(uint256 agentId, uint256 tagIndex, uint256 taskScoreScaled) external onlyAdmin returns (uint256) {
        require(registry.agentExists(agentId), "IncentiveContract: unknown agent");
        require(taskScoreScaled <= SCALE, "IncentiveContract: score > 1.0");

        uint256 oldCap = registry.getCapability(agentId, tagIndex);
        uint256 newCap = (muCap * oldCap + (SCALE - muCap) * taskScoreScaled) / SCALE;

        registry.updateCapability(agentId, tagIndex, newCap);
        emit CapabilityUpdated(agentId, tagIndex, oldCap, newCap, taskScoreScaled);
        return newCap;
    }
}
