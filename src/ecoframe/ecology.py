from __future__ import annotations

import math
from abc import ABC, abstractmethod

from ecoframe.agent import EcoAgent, BrainState
from ecoframe.field import Field
from ecoframe.signal import CuriositySignal
from ecoframe.environment import Environment


class Ecology(Environment, ABC):
    """
    Abstract base for all EcoFrame ecologies.

    Extends Environment so all Ecology subclasses are also Environments.
    Provides _gossip_cycle() and motor_bias() as shared coordination
    infrastructure — no reimplementing gossip in subclasses.

    Interface convention (not enforced by ABC):
      self.agents — iterable of agents with an id or agent_id attribute
      self.field  — object with _backend (A2AField) and step()
    """

    @property
    def capacity(self) -> int:
        """Number of agents. Subclasses should override for precise capacity."""
        agents = getattr(self, 'agents', None) or getattr(self, '_agents', None) or []
        return len(list(agents))

    @abstractmethod
    def step(self) -> dict[str, float]:
        """Run one training iteration. Returns per-agent losses."""

    # ── Shared coordination infrastructure ───────────────────────────────────

    def _gossip_cycle(
        self,
        agent_results: list[tuple[str, float, float]],
        *,
        publish_signals: bool = False,
    ) -> None:
        """
        Complete gossip round: receive → trust → relay → cache → publish → step.

        agent_results : [(agent_id, raw_loss, normalised_surp), ...]
        publish_signals=True  : _gossip_cycle publishes raw_loss (batched path).
        publish_signals=False : agents already published; skip publish here.

        Motor gradients are cached before field.step() wipes _received so
        motor_bias() returns the correct value on the next step.
        """
        a2a = self.field._backend

        for aid, _, _ in agent_results:
            a2a.receive(aid)

        for aid, _, norm_surp in agent_results:
            a2a.update_trust_from_surprise(aid, norm_surp)

        for aid, _, _ in agent_results:
            a2a.relay(aid, threshold=max(0.1, self._surprise_ema_for(aid)))

        if not hasattr(self, '_cached_motor_grads'):
            self._cached_motor_grads: dict[str, tuple[float, float]] = {}
        for aid, _, _ in agent_results:
            self._cached_motor_grads[aid] = a2a.gradient(aid)

        if publish_signals:
            for aid, raw_loss, _ in agent_results:
                if math.isfinite(raw_loss) and raw_loss > 1e-4:
                    agent = self._agent_by_id(aid)
                    pos   = self._agent_pos(agent) if agent else (0.0, 0.0)
                    a2a.publish(aid, CuriositySignal(
                        position=pos, timestamp=a2a._timestamp,
                        publisher=aid, visual=raw_loss,
                    ))

        self.field.step()

    def motor_bias(self, agent_id: str) -> tuple[float, float]:
        """Cached gossip gradient from last _gossip_cycle. (0,0) before first."""
        return getattr(self, '_cached_motor_grads', {}).get(agent_id, (0.0, 0.0))

    def _surprise_ema_for(self, agent_id: str) -> float:
        a = self._agent_by_id(agent_id)
        return getattr(a, '_surprise_ema', 1.0) if a else 1.0

    def _agent_pos(self, agent) -> tuple[float, float]:
        if hasattr(agent, 'body') and hasattr(agent.body, 'pos'):
            p = agent.body.pos
            return (float(p[0]), float(p[2] if len(p) > 2 else p[1]))
        return (0.0, 0.0)

    def _agent_by_id(self, agent_id: str):
        for a in self.agents:
            aid = getattr(a, 'agent_id', None) or getattr(a, 'id', None)
            if aid == agent_id:
                return a
        return None


class SimpleEcology(Ecology):
    """
    Sequential ecology for conventional agents.

    Calls observe / predict / act / surprise on each agent by convention —
    agents don't need to subclass anything, just implement those methods.
    Good for new domains, prototypes, and non-GPU agents.

    Motor bias for step N comes from signals received at the end of step N-1.
    Each agent publishes its own signal after surprise(); _gossip_cycle then
    runs receive/relay/cache/trust/step.
    """

    def __init__(self, world, agents: list, field: Field):
        self._agents = list(agents)
        self._field  = field
        self.world   = world
        self._step   = 0

        for agent in self._agents:
            agent._field = field._backend
            field.register_agent(
                getattr(agent, 'agent_id', None) or getattr(agent, 'id', ''),
                self._agent_pos(agent),
            )

    @property
    def agents(self):
        return self._agents

    @property
    def field(self) -> Field:
        return self._field

    def step(self) -> dict[str, float]:
        state = BrainState(step=self._step)

        for agent in self._agents:
            aid = getattr(agent, 'agent_id', None) or getattr(agent, 'id', '')
            self._field.update_position(aid, self._agent_pos(agent))

        predictions = {}
        for agent in self._agents:
            aid  = getattr(agent, 'agent_id', None) or getattr(agent, 'id', '')
            bias = self.motor_bias(aid)
            obs  = agent.observe(self.world)
            pred = agent.predict(obs)
            act  = agent.act(state)
            if hasattr(act, 'motor_bias'):
                act.motor_bias = (
                    act.motor_bias[0] + 0.15 * bias[0],
                    act.motor_bias[1] + 0.15 * bias[1],
                )
            predictions[aid] = pred

        try:
            self.world.step()
        except AttributeError:
            pass

        agent_results = []
        losses: dict[str, float] = {}
        for agent in self._agents:
            aid    = getattr(agent, 'agent_id', None) or getattr(agent, 'id', '')
            actual = agent.observe(self.world)
            sig    = agent.surprise(predictions[aid], actual)
            sig.validate()

            r = sig.R
            agent._surprise_ema = 0.99 * getattr(agent, '_surprise_ema', 1.0) + 0.01 * max(1e-8, r)
            agent._step_count   = getattr(agent, '_step_count', 0) + 1

            if agent._field is not None and r > 0:
                agent._field.publish(aid, sig)

            losses[aid] = r
            agent_results.append((aid, r, r))

        self._gossip_cycle(agent_results, publish_signals=False)
        self._step += 1
        return losses
