import re

HTTP_RE = re.compile(r"ST@RT.+?INFO\s+(.+?)\s+END", re.MULTILINE | re.DOTALL)


def http_re(data):
    """
    Extracts HTTP requests from raw data string in special logging format.

    Logging format `ST@RT\n%(asctime)s %(levelname)-8s\n%(message)s\nEND`
    where `message` is a required HTTP request bytes.
    """
    return HTTP_RE.findall(data)


def get_requests_from_file(path):
    """
    Reads raw HTTP requests from given file.
    """
    with open(path, 'r') as f:
        file_data = f.read()
    requests = http_re(file_data)
    return requests


def batch_generator(inputs, lengths, num_epochs, batch_size, vocab):
    i = 0
    input_size = len(inputs)
    for _ in range(num_epochs):
        while i + batch_size <= input_size:
            length = lengths[i:i + batch_size]
            padded = batch_padding(inputs[i:i + batch_size], length, vocab)
            yield padded, length
            i += batch_size
        i = 0


def batch_padding(inputs, lengths, vocab):
    max_length = max(lengths)
    return [sample + ([vocab.vocab['<PAD>']] * (max_length - len(sample))) for sample in inputs]


def one_by_one_generator(inputs, lengths):
    """
    data to it length
    """
    for i in range(len(inputs)):
        yield [inputs[i]], lengths[i]


def print_progress(step, epoch, loss, step_loss, time):
    """
    Prints learning stage progress.
    """
    msg = "Step {} (epoch {}), average_train_loss = {:.5f}, step_loss = {:.5f}, time_per_step = {:.3f}"
    msg = msg.format(step, epoch, loss, step_loss, time)
    print(msg)
