"""Sentence-preserving allocation of estimated speech to TaoMate groups."""


def sentence_word_owners(sentences, durations):
    """Assign ordered words once, splitting only sentences longer than 1.5 groups."""
    weights = [weight for sentence in sentences for weight in sentence]
    if not weights or not durations:
        return []
    if any(value <= 0 for value in weights + list(durations)):
        raise ValueError("Dialogue weights and chunk durations must be positive")
    # Preserve the existing shot-wide speaking pace, including opening silence.
    scale = sum(durations) / sum(weights)
    cumulative = [0.0]
    for weight in weights:
        cumulative.append(cumulative[-1] + weight * scale)
    nominal = max(durations)
    cuts = {0, len(weights)}
    sentence_ends = {0}
    start = 0
    for sentence in sentences:
        end = start + len(sentence)
        cuts.add(end)
        sentence_ends.add(end)
        if cumulative[end] - cumulative[start] > 1.5 * nominal:
            cuts.update(range(start + 1, end))
        start = end
    # ponytail: O(groups * candidate cuts squared), suitable for shot dialogue;
    # very long scripts can use a bounded search without changing allowed cuts.
    cuts = sorted(cuts)
    states = {0: (0.0, [])}
    for duration in durations:
        following = {}
        for end in cuts:
            choices = []
            for start, (cost, boundaries) in states.items():
                if start > end:
                    continue
                spoken = cumulative[end] - cumulative[start]
                error = (spoken - duration) ** 2 / duration
                # Prefer complete sentences and avoid silent groups where possible.
                penalty = nominal if start == end else 0.0
                penalty += 0.1 * nominal if end not in sentence_ends else 0.0
                choices.append((cost + error + penalty, boundaries + [end]))
            following[end] = min(choices, key=lambda item: item[0])
        states = following
    owners = []
    for index, end in enumerate(states[len(weights)][1]):
        owners.extend([index] * (end - len(owners)))
    return owners
