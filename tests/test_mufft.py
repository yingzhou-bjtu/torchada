import pytest
import torch

from torchada import musa_irfft_graph


@pytest.mark.musa
def test_musa_irfft_graph_keeps_one_plan_per_stream() -> None:
    n = 8
    spectrum = torch.randn(1, n // 2 + 1, 2, device="musa", dtype=torch.complex64)
    expected = torch.fft.irfft(spectrum.cpu(), n=n, dim=1, norm="backward")
    stream_a = torch.musa.Stream()
    stream_b = torch.musa.Stream()

    with torch.musa.stream(stream_a):
        output_a = musa_irfft_graph(spectrum, n)
    with torch.musa.stream(stream_b):
        output_b = musa_irfft_graph(spectrum, n)

    stream_a.synchronize()
    stream_b.synchronize()
    torch.testing.assert_close(output_a.cpu(), expected)
    torch.testing.assert_close(output_b.cpu(), expected)

    graph_stream = torch.musa.Stream()
    graph = torch.musa.MUSAGraph()
    with torch.musa.stream(graph_stream):
        musa_irfft_graph(spectrum, n)
        graph_stream.synchronize()
        with torch.musa.graph(graph, stream=graph_stream):
            graph_output = musa_irfft_graph(spectrum, n)
    graph.replay()
    graph_stream.synchronize()
    torch.testing.assert_close(graph_output.cpu(), expected)
