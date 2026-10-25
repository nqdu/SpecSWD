"""Run from the repository root: python -m specd.torch.example."""

import torch

from specd.torch import SpecWorkSpace


def main():
    rho = torch.tensor([2.8, 2.8, 3.2], dtype=torch.float64, requires_grad=True)
    vsh = torch.tensor([3.3, 3.3, 5.5], dtype=torch.float64, requires_grad=True)
    vsv = torch.tensor([3.0, 3.0, 5.0], dtype=torch.float64, requires_grad=True)
    workspace = SpecWorkSpace().initialize("love", [0, 35, 35], rho, vsh=vsh, vsv=vsv)
    phase = workspace.compute_egn(0.1)
    group = workspace.group_velocity()
    print("phase velocities:", phase.detach())
    print("group velocities:", group.detach())
    loss = (phase[0].real - 3.5).square()
    loss.backward()
    print("d(loss)/d(vsh):", vsh.grad)
    print("d(loss)/d(vsv):", vsv.grad)
    print("d(loss)/d(rho):", rho.grad)


if __name__ == "__main__":
    main()
