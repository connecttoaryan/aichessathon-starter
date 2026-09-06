"""Pseudo-legal move generation + make + legality filter + perft.

Move encoding (one int):
    from        bits 0..5
    to          bits 6..11
    promo       bits 12..14   (0 none, 1 N, 2 B, 3 R, 4 Q)
    flag        bits 15..16   (0 normal, 1 double push, 2 en-passant, 3 castle)
"""

from .bb_core import (
    BB,
    BK,
    BN,
    BP,
    BQ,
    BR,
    C_BK,
    C_BQ,
    C_WK,
    C_WQ,
    CR,
    EP,
    HM,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    NO_EP,
    PAWN_ATTACKS,
    STM,
    WB,
    WK,
    WN,
    WP,
    WQ,
    WR,
    all_occ,
    bishop_attacks,
    black_occ,
    is_square_attacked,
    lsb,
    queen_attacks,
    rook_attacks,
    white_occ,
)

PROMO_PIECES = (WN, WB, WR, WQ)  # order matches promo codes 1..4 (white)


def enc(frm, to, promo=0, flag=0):
    return frm | (to << 6) | (promo << 12) | (flag << 15)


def move_from(m):
    return m & 63


def move_to(m):
    return (m >> 6) & 63


def move_promo(m):
    return (m >> 12) & 7


def move_flag(m):
    return (m >> 15) & 3


def _iter_bits(bb):
    while bb:
        b = bb & -bb
        yield b.bit_length() - 1
        bb ^= b


def gen_pseudo(bd):
    """Generate pseudo-legal moves for the side to move."""
    moves = []
    white = bd[STM] == 0
    own = white_occ(bd) if white else black_occ(bd)
    opp = black_occ(bd) if white else white_occ(bd)
    occ = own | opp
    empty = ~occ & ((1 << 64) - 1)

    if white:
        pawns, knights, bishops, rooks, queens, king = WP, WN, WB, WR, WQ, WK
        push_dir = 8
        start_rank_mask = 0x000000000000FF00  # rank 2
        promo_rank_mask = 0xFF00000000000000  # rank 8 (destination)
        pawn_att = PAWN_ATTACKS[0]
    else:
        pawns, knights, bishops, rooks, queens, king = BP, BN, BB, BR, BQ, BK
        push_dir = -8
        start_rank_mask = 0x00FF000000000000  # rank 7
        promo_rank_mask = 0x00000000000000FF  # rank 1 (destination)
        pawn_att = PAWN_ATTACKS[1]

    # --- Pawns ---
    for frm in _iter_bits(bd[pawns]):
        # single push
        to = frm + push_dir
        to_bit = 1 << to
        if empty & to_bit:
            if to_bit & promo_rank_mask:
                for p in (4, 3, 2, 1):  # Q,R,B,N
                    moves.append(enc(frm, to, p, 0))
            else:
                moves.append(enc(frm, to, 0, 0))
                # double push
                if (1 << frm) & start_rank_mask:
                    to2 = frm + 2 * push_dir
                    if empty & (1 << to2):
                        moves.append(enc(frm, to2, 0, 1))
        # captures
        caps = pawn_att[frm] & opp
        for to in _iter_bits(caps):
            to_bit = 1 << to
            if to_bit & promo_rank_mask:
                for p in (4, 3, 2, 1):
                    moves.append(enc(frm, to, p, 0))
            else:
                moves.append(enc(frm, to, 0, 0))
        # en passant
        if bd[EP] != NO_EP and pawn_att[frm] & (1 << bd[EP]):
            moves.append(enc(frm, bd[EP], 0, 2))

    # --- Knights ---
    for frm in _iter_bits(bd[knights]):
        targets = KNIGHT_ATTACKS[frm] & ~own
        for to in _iter_bits(targets):
            moves.append(enc(frm, to, 0, 0))

    # --- Bishops ---
    for frm in _iter_bits(bd[bishops]):
        targets = bishop_attacks(frm, occ) & ~own
        for to in _iter_bits(targets):
            moves.append(enc(frm, to, 0, 0))

    # --- Rooks ---
    for frm in _iter_bits(bd[rooks]):
        targets = rook_attacks(frm, occ) & ~own
        for to in _iter_bits(targets):
            moves.append(enc(frm, to, 0, 0))

    # --- Queens ---
    for frm in _iter_bits(bd[queens]):
        targets = queen_attacks(frm, occ) & ~own
        for to in _iter_bits(targets):
            moves.append(enc(frm, to, 0, 0))

    # --- King (non-castling) ---
    ksq = lsb(bd[king])
    for to in _iter_bits(KING_ATTACKS[ksq] & ~own):
        moves.append(enc(ksq, to, 0, 0))

    # --- Castling ---
    # Squares: white e1=4,f1=5,g1=6,d1=3,c1=2,b1=1 ; black e8=60,f8=61,g8=62,d8=59,c8=58,b8=57
    if white:
        if (
            (bd[CR] & C_WK)
            and not (occ & ((1 << 5) | (1 << 6)))
            and (
                not is_square_attacked(bd, 4, False)
                and not is_square_attacked(bd, 5, False)
                and not is_square_attacked(bd, 6, False)
            )
        ):
            moves.append(enc(4, 6, 0, 3))
        if (
            (bd[CR] & C_WQ)
            and not (occ & ((1 << 1) | (1 << 2) | (1 << 3)))
            and (
                not is_square_attacked(bd, 4, False)
                and not is_square_attacked(bd, 3, False)
                and not is_square_attacked(bd, 2, False)
            )
        ):
            moves.append(enc(4, 2, 0, 3))
    else:
        if (
            (bd[CR] & C_BK)
            and not (occ & ((1 << 61) | (1 << 62)))
            and (
                not is_square_attacked(bd, 60, True)
                and not is_square_attacked(bd, 61, True)
                and not is_square_attacked(bd, 62, True)
            )
        ):
            moves.append(enc(60, 62, 0, 3))
        if (
            (bd[CR] & C_BQ)
            and not (occ & ((1 << 57) | (1 << 58) | (1 << 59)))
            and (
                not is_square_attacked(bd, 60, True)
                and not is_square_attacked(bd, 59, True)
                and not is_square_attacked(bd, 58, True)
            )
        ):
            moves.append(enc(60, 58, 0, 3))

    return moves


_WHITE_PIECES = (WP, WN, WB, WR, WQ, WK)
_BLACK_PIECES = (BP, BN, BB, BR, BQ, BK)


def make(bd, m):
    """Return a new board array after applying move m. Assumes m is pseudo-legal."""
    nb = bd[:]  # copy
    white = bd[STM] == 0
    frm = move_from(m)
    to = move_to(m)
    promo = move_promo(m)
    flag = move_flag(m)
    frm_bit = 1 << frm
    to_bit = 1 << to

    own = _WHITE_PIECES if white else _BLACK_PIECES
    opp = _BLACK_PIECES if white else _WHITE_PIECES

    # Which of our pieces moved?
    moved = -1
    for pi in own:
        if nb[pi] & frm_bit:
            moved = pi
            break

    is_pawn = moved == (WP if white else BP)
    is_capture = (all_occ(bd) & to_bit) != 0 or flag == 2

    # Remove moving piece from origin.
    nb[moved] ^= frm_bit

    # Handle capture on destination (normal).
    if flag != 2 and (to_bit & (black_occ(bd) if white else white_occ(bd))):
        for pi in opp:
            if nb[pi] & to_bit:
                nb[pi] ^= to_bit
                break

    # En-passant capture removes the pawn behind `to`.
    if flag == 2:
        cap_sq = to - 8 if white else to + 8
        cap_bit = 1 << cap_sq
        nb[BP if white else WP] ^= cap_bit

    # Place moving piece (with promotion).
    if promo:
        promo_idx = PROMO_PIECES[promo - 1]
        if not white:
            promo_idx += 6
        nb[promo_idx] |= to_bit
    else:
        nb[moved] |= to_bit

    # Castling: move the rook.
    if flag == 3:
        if to == 6:  # white kingside e1g1, rook h1->f1
            nb[WR] ^= 1 << 7
            nb[WR] |= 1 << 5
        elif to == 2:  # white queenside e1c1, rook a1->d1
            nb[WR] ^= 1 << 0
            nb[WR] |= 1 << 3
        elif to == 62:  # black kingside
            nb[BR] ^= 1 << 63
            nb[BR] |= 1 << 61
        elif to == 58:  # black queenside
            nb[BR] ^= 1 << 56
            nb[BR] |= 1 << 59

    # Update castling rights.
    cr = nb[CR]
    if moved == WK:
        cr &= ~(C_WK | C_WQ)
    elif moved == BK:
        cr &= ~(C_BK | C_BQ)
    # Rook moved off its home square.
    if frm == 0 or to == 0:
        cr &= ~C_WQ
    if frm == 7 or to == 7:
        cr &= ~C_WK
    if frm == 56 or to == 56:
        cr &= ~C_BQ
    if frm == 63 or to == 63:
        cr &= ~C_BK
    nb[CR] = cr

    # En-passant target.
    if flag == 1:  # double push
        nb[EP] = frm + (8 if white else -8)
    else:
        nb[EP] = NO_EP

    # Halfmove clock.
    if is_pawn or is_capture:
        nb[HM] = 0
    else:
        nb[HM] = bd[HM] + 1

    nb[STM] = 1 - bd[STM]
    return nb


def king_sq(bd, white):
    return lsb(bd[WK] if white else bd[BK])


def in_check(bd, white):
    return is_square_attacked(bd, king_sq(bd, white), not white)


def gen_legal(bd):
    """Filter pseudo-legal moves: the mover's king must not be in check after."""
    white = bd[STM] == 0
    legal = []
    for m in gen_pseudo(bd):
        nb = make(bd, m)
        if not is_square_attacked(nb, king_sq(nb, white), not white):
            legal.append(m)
    return legal


def perft(bd, depth):
    if depth == 0:
        return 1
    total = 0
    for m in gen_legal(bd):
        total += perft(make(bd, m), depth - 1)
    return total
