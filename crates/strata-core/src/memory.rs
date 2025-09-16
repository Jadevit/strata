//! Lightweight rolling, per-session memory of chat turns.
//! Long-term memory can live here later; for now we only keep the running dialog.

use crate::traits::backend::{ChatTurn, Role};

#[derive(Default, Debug, Clone)]
pub struct SessionMemory {
    turns: Vec<ChatTurn>,
}

impl SessionMemory {
    #[inline]
    pub fn new() -> Self {
        Self { turns: Vec::new() }
    }

    /// All stored turns (oldest → newest).
    #[inline]
    pub fn turns(&self) -> &[ChatTurn] {
        &self.turns
    }

    /// Push a new user turn.
    #[inline]
    pub fn push_user<S: Into<String>>(&mut self, s: S) {
        self.turns.push(ChatTurn {
            role: Role::User,
            content: s.into(),
        });
    }

    /// Push a new assistant turn.
    #[inline]
    pub fn push_assistant<S: Into<String>>(&mut self, s: S) {
        self.turns.push(ChatTurn {
            role: Role::Assistant,
            content: s.into(),
        });
    }

    /// Push a system turn (rare mid-session; usually set at engine-level).
    #[inline]
    pub fn push_system<S: Into<String>>(&mut self, s: S) {
        self.turns.push(ChatTurn {
            role: Role::System,
            content: s.into(),
        });
    }

    /// Remove all history.
    #[inline]
    pub fn clear(&mut self) {
        self.turns.clear();
    }

    /// Drop the oldest non-system turn(s) to make room.
    /// If the oldest is a (User, Assistant) pair, remove them together
    /// to keep dialog coherent. Returns true if something was removed.
    pub fn drop_oldest_pair(&mut self) -> bool {
        if self.turns.is_empty() {
            return false;
        }

        // Find first non-system turn.
        let Some(i) = self
            .turns
            .iter()
            .position(|t| !matches!(t.role, Role::System))
        else {
            return false;
        };

        // Prefer dropping a coherent (User, Assistant) pair if present.
        if i + 1 < self.turns.len()
            && matches!(self.turns[i].role, Role::User)
            && matches!(self.turns[i + 1].role, Role::Assistant)
        {
            self.turns.drain(i..=i + 1);
        } else {
            self.turns.remove(i);
        }
        true
    }
}
