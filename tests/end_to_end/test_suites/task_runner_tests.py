# Copyright 2020-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import logging

from tests.end_to_end.utils.common_fixtures import (
    fx_federation_tr,
    fx_federation_tr_dws,
)
from tests.end_to_end.utils import federation_helper as fed_helper

log = logging.getLogger(__name__)


@pytest.mark.task_runner_basic
def test_federation_via_native(request, fx_federation_tr):
    """
    Test federation via native task runner.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr (Fixture): Pytest fixture for native task runner
    """
    # Start the federation
    results = fed_helper.run_federation(fx_federation_tr)

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr,
        results,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"


@pytest.mark.task_runner_dockerized_ws
def test_federation_via_dockerized_workspace(request, fx_federation_tr_dws):
    """
    Test federation via dockerized workspace.
    Args:
        request (Fixture): Pytest fixture
        fx_federation_tr_dws (Fixture): Pytest fixture for dockerized workspace
    """
    # Start the federation
    results = fed_helper.run_federation_for_dws(
        fx_federation_tr_dws, use_tls=request.config.use_tls
    )

    # Verify the completion of the federation run
    assert fed_helper.verify_federation_run_completion(
        fx_federation_tr_dws,
        results,
        test_env=request.config.test_env,
        num_rounds=request.config.num_rounds,
    ), "Federation completion failed"
