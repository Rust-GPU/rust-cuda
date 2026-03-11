#![allow(bindings_with_variant_name)] // for using c on AsciiChar

use crate::types::*;
use ascii::{AsciiChar, AsciiStr, Chars};
use std::convert::TryFrom;
use std::str::FromStr;
use std::{iter::Peekable, ops::Range};

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub range: Range<usize>,
}

pub type LexerResult = Result<Token, String>;

/// A lexer for the PTX ISA. Yields tokens through an [`Iterator`] implementation.
///
/// According to the [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#source-format), source files
/// are ASCII, therefore we use [`AsciiStr`] for performance (avoiding utf8 decoding).
#[derive(Debug)]
pub struct Lexer<'src> {
    pub src: &'src AsciiStr,
    cur: usize,
    iter: Peekable<Chars<'src>>,
    // hack used for dealing with .0 modifiers being either options or
    // numbers
    prev_was_assign: bool,
    brace_stack: u32,
    pub values: Vec<Option<TokenValue>>,
}

fn is_ident_continue(c: AsciiChar) -> bool {
    c.is_ascii_alphanumeric() || c == AsciiChar::UnderScore || c == AsciiChar::Dollar
}

fn is_ident_start(c: AsciiChar) -> bool {
    c.is_ascii_alphabetic()
        || c == AsciiChar::UnderScore
        || c == AsciiChar::Dollar
        || c == AsciiChar::Percent
}

impl Iterator for Lexer<'_> {
    type Item = LexerResult;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src AsciiStr) -> Self {
        let mut iter = src.chars().peekable();
        iter.next();
        Self {
            src,
            cur: 0,
            iter,
            prev_was_assign: false,
            brace_stack: 0,
            values: vec![],
        }
    }

    fn next(&mut self) -> Option<AsciiChar> {
        let next = self.iter.next();
        self.cur += 1;
        next
    }

    fn peek(&mut self) -> Option<AsciiChar> {
        self.iter.peek().copied()
    }

    fn cur_char(&self) -> AsciiChar {
        self.src[self.cur]
    }

    fn eat_until(&mut self, f: impl Fn(AsciiChar, Option<AsciiChar>) -> bool) -> &AsciiStr {
        let cur = self.cur;
        while let Some(c) = self.next() {
            if !f(c, self.peek()) {
                break;
            }
        }
        let now = self.cur;
        &self.src[cur..now]
    }

    fn eat_and_ret_token(&mut self, len: usize, kind: TokenKind) -> Token {
        let cur = self.cur;
        for _ in 0..len {
            self.next();
        }
        Token {
            kind,
            range: cur..self.cur,
        }
    }

    fn next_token(&mut self) -> Option<LexerResult> {
        if self.cur == self.src.len() {
            return None;
        }
        self.values.push(None);
        let cur = self.cur_char();
        Some(Ok(match cur {
            AsciiChar::Percent => {
                let peek = self.peek();
                if let Some(peeked) = peek
                    && is_ident_continue(peeked)
                {
                    return Some(Ok(self.opcode_or_ident()));
                }
                self.eat_and_ret_token(1, TokenKind::Modulo)
            }
            c if is_ident_start(c) => self.opcode_or_ident(),
            c if c.is_ascii_digit() => return Some(self.number()),
            AsciiChar::Quotation => return Some(self.string()),
            AsciiChar::Dot => return Some(self.dot()),
            AsciiChar::Tilde => self.eat_and_ret_token(1, TokenKind::Tilde),
            AsciiChar::Asterisk => self.eat_and_ret_token(1, TokenKind::Times),
            AsciiChar::Minus => self.eat_and_ret_token(1, TokenKind::Minus),
            AsciiChar::Plus => self.eat_and_ret_token(1, TokenKind::Plus),
            AsciiChar::Comma => self.eat_and_ret_token(1, TokenKind::Comma),
            AsciiChar::At => self.eat_and_ret_token(1, TokenKind::At),
            AsciiChar::BracketOpen => self.eat_and_ret_token(1, TokenKind::SquareBracketOpen),
            AsciiChar::BracketClose => self.eat_and_ret_token(1, TokenKind::SquareBracketClose),
            AsciiChar::ParenOpen => self.eat_and_ret_token(1, TokenKind::ParenOpen),
            AsciiChar::ParenClose => self.eat_and_ret_token(1, TokenKind::ParenClose),
            AsciiChar::Colon => self.eat_and_ret_token(1, TokenKind::Colon),
            AsciiChar::Semicolon => self.eat_and_ret_token(1, TokenKind::Semicolon),
            AsciiChar::CurlyBraceClose => {
                if self.prev_was_assign {
                    self.brace_stack = self.brace_stack.saturating_sub(1);
                    if self.brace_stack == 0 {
                        self.prev_was_assign = false;
                    }
                }
                self.eat_and_ret_token(1, TokenKind::CurlyBracketClose)
            }
            AsciiChar::CurlyBraceOpen => {
                if self.prev_was_assign {
                    self.brace_stack += 1;
                }
                self.eat_and_ret_token(1, TokenKind::CurlyBracketOpen)
            }
            AsciiChar::Caret => self.eat_and_ret_token(1, TokenKind::Xor),
            AsciiChar::Question => self.eat_and_ret_token(1, TokenKind::QuestionMark),
            AsciiChar::Ampersand if self.peek() == Some(AsciiChar::Ampersand) => {
                self.eat_and_ret_token(2, TokenKind::LogicalAnd)
            }
            AsciiChar::Ampersand => self.eat_and_ret_token(1, TokenKind::And),
            AsciiChar::VerticalBar if self.peek() == Some(AsciiChar::VerticalBar) => {
                self.eat_and_ret_token(2, TokenKind::LogicalOr)
            }
            AsciiChar::VerticalBar => self.eat_and_ret_token(1, TokenKind::Or),
            AsciiChar::Equal if self.peek() == Some(AsciiChar::Equal) => {
                let cur = self.cur;
                self.next();
                self.next();
                Token {
                    kind: TokenKind::Equals,
                    range: cur..self.cur,
                }
            }
            AsciiChar::Equal => {
                self.prev_was_assign = true;
                self.eat_and_ret_token(1, TokenKind::Assign)
            }
            AsciiChar::Exclamation => {
                let cur = self.cur;
                let next = self.next();
                let kind = match next {
                    Some(AsciiChar::Equal) => {
                        self.next();
                        TokenKind::NotEquals
                    }
                    _ => TokenKind::Bang,
                };
                Token {
                    kind,
                    range: cur..self.cur,
                }
            }
            AsciiChar::LessThan => {
                let cur = self.cur;
                let next = self.next();
                let kind = match next {
                    Some(AsciiChar::LessThan) => {
                        self.next();
                        TokenKind::LeftShift
                    }
                    Some(AsciiChar::Equal) => {
                        self.next();
                        TokenKind::LessThanOrEqualTo
                    }
                    _ => TokenKind::LessThan,
                };
                Token {
                    kind,
                    range: cur..self.cur,
                }
            }
            AsciiChar::GreaterThan => {
                let cur = self.cur;
                let next = self.next();
                let kind = match next {
                    Some(AsciiChar::GreaterThan) => {
                        self.next();
                        TokenKind::RightShift
                    }
                    Some(AsciiChar::Equal) => {
                        self.next();
                        TokenKind::GreaterThanOrEqualTo
                    }
                    _ => TokenKind::GreaterThan,
                };
                Token {
                    kind,
                    range: cur..self.cur,
                }
            }
            AsciiChar::LineFeed => {
                self.next();
                return self.next_token();
            }
            AsciiChar::Slash => {
                let next = self.peek();
                match next {
                    Some(AsciiChar::Slash) => {
                        self.next();
                        self.eat_until(|c, _| {
                            c != AsciiChar::LineFeed && c != AsciiChar::CarriageReturn
                        });
                        if self.cur != self.src.len()
                            && self.cur_char() == AsciiChar::CarriageReturn
                            && self.peek() == Some(AsciiChar::LineFeed)
                        {
                            self.next();
                        }

                        return self.next_token();
                    }
                    Some(AsciiChar::Asterisk) => {
                        self.next();
                        self.eat_until(|c, next| {
                            c == AsciiChar::Asterisk && next == Some(AsciiChar::Slash)
                        });
                        self.next();
                        return self.next_token();
                    }
                    _ => self.eat_and_ret_token(1, TokenKind::Divide),
                }
            }
            AsciiChar::Space | AsciiChar::Tab => {
                self.next();
                return self.next_token();
            }
            c => {
                self.next();
                return Some(Err(format!("Unexpected token `{c}`")));
            }
        }))
    }

    fn string(&mut self) -> LexerResult {
        let cur = self.cur;
        loop {
            match self.next() {
                None => return Err("Unterminated string literal".to_string()),
                Some(AsciiChar::BackSlash) => {
                    self.next();
                }
                Some(AsciiChar::Quotation) => {
                    self.next();
                    return Ok(Token {
                        kind: TokenKind::String,
                        range: cur..self.cur,
                    });
                }
                _ => {}
            }
        }
    }

    fn number(&mut self) -> LexerResult {
        let cur = self.cur;

        let val = if self.cur_char() == AsciiChar::Dot {
            self.eat_until(|c, _| c.is_ascii_digit());
            let string = &self.src[cur..self.cur];
            let val = string
                .as_str()
                .parse::<f64>()
                .map_err(|_| format!("Failed to parse `{string}` as f64 literal"))?;

            *self.values.last_mut().unwrap() = Some(TokenValue::Double(val));
            return Ok(Token {
                kind: TokenKind::Double,
                range: cur..self.cur,
            });
        } else if self.cur_char() != AsciiChar::_0 {
            let mut is_double = false;
            while let Some(next) = self.next() {
                match next {
                    c if c.is_ascii_digit() => {}
                    AsciiChar::Dot => is_double = true,
                    _ => break,
                }
            }

            let string = &self.src[cur..self.cur];
            if is_double {
                let val = string
                    .as_str()
                    .parse::<f64>()
                    .map_err(|_| format!("Failed to parse `{string}` as f64 literal"))?;

                *self.values.last_mut().unwrap() = Some(TokenValue::Double(val));
                return Ok(Token {
                    kind: TokenKind::Double,
                    range: cur..self.cur,
                });
            }

            string
                .as_str()
                .parse()
                .map_err(|_| "Failed to parse hex literal as u64".to_string())?
        } else {
            let next = self.next();
            match next {
                None => 0,
                Some(AsciiChar::Dot) => {
                    self.eat_until(|c, _| c.is_ascii_digit());
                    let string = &self.src[cur..self.cur];
                    let val = string
                        .as_str()
                        .parse::<f64>()
                        .map_err(|_| format!("Failed to parse `{string}` as f64 literal"))?;

                    *self.values.last_mut().unwrap() = Some(TokenValue::Double(val));
                    return Ok(Token {
                        kind: TokenKind::Double,
                        range: cur..self.cur,
                    });
                }
                Some(AsciiChar::x | AsciiChar::X) => {
                    self.next();
                    let num = self.eat_until(|c, _| c.is_ascii_hexdigit());
                    u64::from_str_radix(num.as_str(), 16)
                        .map_err(|_| "Failed to parse hex literal as u64".to_string())?
                }
                Some(AsciiChar::b | AsciiChar::B) => {
                    self.next();
                    let num = self.eat_until(|c, _| c == AsciiChar::_0 || c == AsciiChar::_1);
                    u64::from_str_radix(num.as_str(), 2)
                        .map_err(|_| "Failed to parse binary literal as u64".to_string())?
                }
                Some(AsciiChar::f | AsciiChar::F) => {
                    self.next();
                    if (self.src.len() - self.cur) < 8 {
                        return Err(format!("Expected 8 numbers after `0f` at `{}`", self.cur));
                    }
                    let numbers = &self.src[self.cur..self.cur + 8];
                    for _ in 0..8 {
                        self.next();
                    }

                    let raw = u32::from_str_radix(numbers.as_str(), 16).map_err(|_| {
                        format!("Failed to parse `{numbers}` as a 32 bit hex integer")
                    })?;

                    *self.values.last_mut().unwrap() = Some(TokenValue::Float(f32::from_bits(raw)));
                    return Ok(Token {
                        kind: TokenKind::Float,
                        range: cur..self.cur,
                    });
                }

                Some(AsciiChar::d | AsciiChar::D) => {
                    self.next();
                    if (self.src.len() - self.cur) < 16 {
                        return Err(format!("Expected 16 numbers after `0d` at `{}`", self.cur));
                    }
                    let numbers = &self.src[self.cur..self.cur + 8];
                    for _ in 0..16 {
                        self.next();
                    }

                    let raw = u64::from_str_radix(numbers.as_str(), 16).map_err(|_| {
                        format!("Failed to parse `{numbers}` as a 64 bit hex integer")
                    })?;

                    *self.values.last_mut().unwrap() =
                        Some(TokenValue::Double(f64::from_bits(raw)));
                    return Ok(Token {
                        kind: TokenKind::Double,
                        range: cur..self.cur,
                    });
                }
                Some(c) if c.as_byte() >= b'0' && c.as_byte() <= b'7' => {
                    let num = self.eat_until(|c, _| c.as_byte() >= b'0' && c.as_byte() <= b'7');
                    u64::from_str_radix(num.as_str(), 8)
                        .map_err(|_| "Failed to parse octal literal as u64".to_string())?
                }
                Some(_) => 0,
            }
        };

        // per the spec:
        //
        // Integer literals are non-negative and have a type determined by their magnitude and
        // optional type suffix as follows: literals are signed (.s64) unless
        // the value cannot be fully represented in .s64 or the unsigned suffix is specified,
        // in which case the literal is unsigned (.u64).
        let is_signed = if self.src.as_bytes().get(self.cur) == Some(&b'U') {
            self.next();
            false
        } else {
            i64::try_from(val).is_ok()
        };

        let kind = if is_signed {
            *self.values.last_mut().unwrap() = Some(TokenValue::SignedInt(val as i64));
            TokenKind::SignedInt
        } else {
            *self.values.last_mut().unwrap() = Some(TokenValue::UnsignedInt(val));
            TokenKind::UnsignedInt
        };

        Ok(Token {
            kind,
            range: cur..self.cur,
        })
    }

    fn opcode_or_ident(&mut self) -> Token {
        let cur = self.cur;
        let ident = self.eat_until(|c, _| is_ident_continue(c));
        // check if its an instruction
        if ident.chars().all(|c| c.is_ascii_alphanumeric())
            && let Ok(kind) = InstructionKind::from_str(ident.as_str())
        {
            *self.values.last_mut().unwrap() = Some(TokenValue::Instruction(kind));
            return Token {
                kind: TokenKind::Instruction,
                range: cur..self.cur,
            };
        }

        *self.values.last_mut().unwrap() =
            Some(TokenValue::Ident(self.src[cur..self.cur].to_string()));
        Token {
            kind: TokenKind::Ident,
            range: cur..self.cur,
        }
    }

    // tries to lex something starting with a dot.
    // it could be either a directive, an option, a type, a dot, or an error.
    fn dot(&mut self) -> LexerResult {
        let cur = self.cur;
        let next = self.next();
        match next {
            None => {
                return Ok(Token {
                    kind: TokenKind::Dot,
                    range: cur..self.cur,
                });
            }
            Some(c) if !is_ident_continue(c) => {
                return Ok(Token {
                    kind: TokenKind::Dot,
                    range: cur..self.cur,
                });
            }
            Some(_) => {}
        }
        let prev_was_assign = self.prev_was_assign;
        let ident = self.eat_until(|c, _| is_ident_continue(c)).to_string();
        if ident
            .chars()
            .all(|c| c.is_lowercase() || c.is_ascii_digit() || c == AsciiChar::UnderScore)
        {
            if ident.chars().all(|c| c.is_ascii_digit()) {
                let string = &self.src[cur..self.cur];

                if string == ".0" && prev_was_assign {
                    *self.values.last_mut().unwrap() =
                        Some(TokenValue::Option(InstructionOption::Dim0));
                    return Ok(Token {
                        kind: TokenKind::Option,
                        range: cur..self.cur,
                    });
                }
                if string == ".1" && prev_was_assign {
                    *self.values.last_mut().unwrap() =
                        Some(TokenValue::Option(InstructionOption::Dim1));
                    return Ok(Token {
                        kind: TokenKind::Option,
                        range: cur..self.cur,
                    });
                }
                if string == ".2" && prev_was_assign {
                    *self.values.last_mut().unwrap() =
                        Some(TokenValue::Option(InstructionOption::Dim2));
                    return Ok(Token {
                        kind: TokenKind::Option,
                        range: cur..self.cur,
                    });
                }

                if let Ok(val) = string.as_str().parse::<f64>() {
                    *self.values.last_mut().unwrap() = Some(TokenValue::Double(val));
                    return Ok(Token {
                        kind: TokenKind::Double,
                        range: cur..self.cur,
                    });
                }
            }

            // simple case, this is a directive
            if let Ok(kind) = DirectiveKind::from_str(ident.as_str()) {
                *self.values.last_mut().unwrap() = Some(TokenValue::Directive(kind));
                return Ok(Token {
                    kind: TokenKind::Directive,
                    range: cur..self.cur,
                });
            }
            if let Ok(kind) = ReservedType::from_str(ident.as_str()) {
                *self.values.last_mut().unwrap() = Some(TokenValue::Type(kind));
                return Ok(Token {
                    kind: TokenKind::Type,
                    range: cur..self.cur,
                });
            }
            if let Ok(kind) = InstructionOption::from_str(ident.as_str()) {
                *self.values.last_mut().unwrap() = Some(TokenValue::Option(kind));
                return Ok(Token {
                    kind: TokenKind::Option,
                    range: cur..self.cur,
                });
            }

            // debug info sections seem to need a little bit of special handling because
            // dwarf sections such as `.debug_info` have a dot before them.
            if ident.starts_with("debug") {
                *self.values.last_mut().unwrap() =
                    Some(TokenValue::Ident(self.src[cur..self.cur].to_string()));
                return Ok(Token {
                    kind: TokenKind::Ident,
                    range: cur..self.cur,
                });
            }
        }

        Err(format!(
            "Expected directive or reserved type, but found `.{ident}` instead"
        ))
    }
}
