/// Wrapper for a model token (ID). Using a newtype avoids accidental
/// mixing with unrelated `i32`s and keeps conversions explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Token(pub i32);

// Using i32 matches llama.cpp's `llama_token`. If a backend uses u32, convert
// at the glue layer and keep this type consistent in core.

impl From<i32> for Token {
    #[inline]
    fn from(value: i32) -> Self {
        Token(value)
    }
}

impl From<Token> for i32 {
    #[inline]
    fn from(token: Token) -> i32 {
        token.0
    }
}
