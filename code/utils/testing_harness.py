# Define all known ground truths in the form of unit tests
def unit_test(R: dict) -> bool:
    known_positive_tickets = ['INC4956642']
    for ticket in known_positive_tickets:
        if ticket in list(R.keys()):
            assert 'yes' in R[ticket][:5].lower() and not 'no' in R[ticket][:5].lower(), 'Violation for ticket {0}'.format(ticket)
        else:
            continue
    return True