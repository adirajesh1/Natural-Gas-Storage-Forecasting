import pandas as pd
import pytest
import requests

from power_forecast.data.eia930 import normalize_eia930
from power_forecast.data.ercot import (
    ErcotApiClient,
    _payload_records,
    normalize_actual_load,
    normalize_adequacy,
    normalize_component_product,
    normalize_outages,
)


def test_current_ercot_payload_schema_is_converted_to_records():
    payload = {
        "fields": [{"name": "postedDatetime"}, {"name": "systemTotal"}],
        "data": [["2026-07-13T12:00:00", 72_000.0]],
    }
    assert _payload_records(payload) == [
        {"postedDatetime": "2026-07-13T12:00:00", "systemTotal": 72_000.0}
    ]


def test_ercot_client_retries_429_and_paginates_current_schema(monkeypatch):
    class Response:
        def __init__(self, status_code, payload, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

    responses = iter(
        [
            Response(429, {}, {"Retry-After": "0"}),
            Response(
                200,
                {"artifacts": [{"_links": {"endpoint": {"href": "https://records"}}}]},
            ),
            Response(
                200,
                {
                    "fields": [{"name": "value"}],
                    "data": [[1]],
                    "_meta": {"currentPage": 1, "totalPages": 2},
                },
            ),
            Response(
                200,
                {
                    "fields": [{"name": "value"}],
                    "data": [[2]],
                    "_meta": {"currentPage": 2, "totalPages": 2},
                },
            ),
        ]
    )
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs.get("params")))
        return next(responses)

    monkeypatch.setattr(requests, "get", fake_get)
    client = ErcotApiClient(
        subscription_key="key",
        id_token="token",
        min_request_interval=0,
    )
    frame = client.fetch_product_records("NP0-000-CD", params={"size": 1})

    assert frame["value"].tolist() == [1, 2]
    assert calls[-1][1]["page"] == 2


@pytest.mark.parametrize(
    ("component", "forecast_column", "product_id"),
    [
        ("wind", "STWPF", "NP4-732-CD"),
        ("solar", "STPPF", "NP4-745-CD"),
    ],
)
def test_renewable_parsers_preserve_forecast_and_actual(component, forecast_column, product_id):
    frame = pd.DataFrame(
        {
            "valid_at": ["2026-07-13T01:00:00Z"],
            "issued_at": ["2026-07-13T00:00:00Z"],
            forecast_column: [12_000.0],
            "GEN": [11_500.0],
        }
    )
    result = normalize_component_product(
        frame,
        component=component,
        product_id=product_id,
        retrieved_at="2026-07-13T00:01:00Z",
    )
    assert result.loc[0, "baseline_mw"] == pytest.approx(12_000.0)
    assert result.loc[0, "actual_mw"] == pytest.approx(11_500.0)


def test_component_parser_accepts_schema_aliases_and_filters_system_rows():
    frame = pd.DataFrame(
        {
            "Delivery Date": ["2026-07-13", "2026-07-13"],
            "Hour Ending": [1, 1],
            "Forecast Zone": ["ERCOT", "NORTH"],
            "Load Forecast": [70_000.0, 20_000.0],
            "Posted Datetime": ["2026-07-12T22:00:00Z"] * 2,
        }
    )
    result = normalize_component_product(
        frame,
        component="load",
        product_id="NP3-560-CD",
        retrieved_at="2026-07-12T22:01:00Z",
    )
    assert len(result) == 1
    assert result.loc[0, "baseline_mw"] == pytest.approx(70_000.0)
    assert str(result.loc[0, "valid_at"].tz) == "UTC"


def test_component_parser_accepts_live_api_hour_ending_text():
    frame = pd.DataFrame(
        {
            "deliveryDate": ["2026-07-13", "2026-07-13"],
            "hourEnding": ["1:00", "24:00"],
            "systemTotal": [70_000.0, 72_000.0],
            "postedDatetime": ["2026-07-12T22:00:00Z"] * 2,
        }
    )
    result = normalize_component_product(
        frame,
        component="load",
        product_id="NP3-560-CD",
        retrieved_at="2026-07-13T01:00:00Z",
    )
    assert result["valid_at"].dt.hour.tolist() == [6, 5]


def test_offset_free_posted_datetime_is_interpreted_as_ercot_local_time():
    frame = pd.DataFrame(
        {
            "deliveryDate": ["2026-07-14"],
            "hourEnding": ["1:00"],
            "systemTotal": [70_000.0],
            "postedDatetime": ["2026-07-13T22:00:00"],
        }
    )
    result = normalize_component_product(
        frame,
        component="load",
        product_id="NP3-560-CD",
        retrieved_at="2026-07-14T04:00:00Z",
    )
    assert result.loc[0, "issued_at"] == pd.Timestamp("2026-07-14T03:00:00Z")


def test_mixed_dst_offsets_in_posted_datetimes_are_normalized_to_utc():
    frame = pd.DataFrame(
        {
            "valid_at": ["2026-11-01T08:00:00Z", "2026-11-01T09:00:00Z"],
            "postedDatetime": [
                "2026-11-01T01:30:00-05:00",
                "2026-11-01T01:30:00-06:00",
            ],
            "systemTotal": [50_000.0, 49_000.0],
        }
    )

    result = normalize_component_product(
        frame,
        component="load",
        product_id="NP3-560-CD",
        retrieved_at="2026-11-01T08:00:00Z",
    )

    assert result["issued_at"].tolist() == [
        pd.Timestamp("2026-11-01T06:30:00Z"),
        pd.Timestamp("2026-11-01T07:30:00Z"),
    ]


def test_regional_rows_are_summed_when_product_has_no_system_total():
    frame = pd.DataFrame(
        {
            "valid_at": ["2026-07-13T01:00:00Z"] * 2,
            "issued_at": ["2026-07-13T00:00:00Z"] * 2,
            "Forecast Zone": ["NORTH", "SOUTH"],
            "baseline_mw": [30_000.0, 20_000.0],
        }
    )
    result = normalize_component_product(
        frame,
        component="load",
        product_id="NP3-560-CD",
        retrieved_at="2026-07-13T00:01:00Z",
    )
    assert len(result) == 1
    assert result.loc[0, "baseline_mw"] == pytest.approx(50_000.0)


def test_actual_outage_and_adequacy_parsers_emit_canonical_values():
    common = {
        "valid_at": ["2026-07-13T01:00:00Z"],
        "issued_at": ["2026-07-13T00:00:00Z"],
    }
    actual = normalize_actual_load(
        pd.DataFrame({**common, "Actual Load": [71_000.0]}),
        retrieved_at="2026-07-13T00:01:00Z",
    )
    outages = normalize_outages(
        pd.DataFrame({**common, "Total Resource Out MW": [4_000.0], "IRR Out MW": [500.0]}),
        retrieved_at="2026-07-13T00:01:00Z",
    )
    adequacy = normalize_adequacy(
        pd.DataFrame({**common, "Total Cap Gen": [90_000.0]}),
        retrieved_at="2026-07-13T00:01:00Z",
    )
    assert actual.loc[0, "load_actual_mw"] == pytest.approx(71_000.0)
    assert outages.loc[0, "conventional_outage_mw"] == pytest.approx(4_000.0)
    assert adequacy.loc[0, "available_capacity_mw"] == pytest.approx(90_000.0)


def test_capacity_parser_rejects_unrecognized_value_schema():
    frame = pd.DataFrame(
        {
            "valid_at": ["2026-07-13T01:00:00Z"],
            "issued_at": ["2026-07-13T00:00:00Z"],
            "renamedCapacityField": [90_000.0],
        }
    )

    with pytest.raises(ValueError, match="no recognized capacity values"):
        normalize_adequacy(frame, retrieved_at="2026-07-13T00:01:00Z")


def test_live_outage_schema_sums_zone_columns():
    frame = pd.DataFrame(
        {
            "postedDatetime": ["2026-07-13T00:00:00Z"],
            "operatingDate": ["2026-07-13"],
            "hourEnding": ["1:00"],
            "totalResourceMWZoneSouth": [1_000.0],
            "totalResourceMWZoneNorth": [2_000.0],
            "totalIRRMWZoneSouth": [100.0],
            "totalIRRMWZoneNorth": [200.0],
            "totalNewEquipResourceMWZoneSouth": [10.0],
            "totalNewEquipResourceMWZoneNorth": [20.0],
        }
    )
    result = normalize_outages(frame, retrieved_at="2026-07-13T01:00:00Z")
    assert result.loc[0, "conventional_outage_mw"] == pytest.approx(3_000.0)
    assert result.loc[0, "renewable_outage_mw"] == pytest.approx(300.0)
    assert result.loc[0, "new_equipment_outage_mw"] == pytest.approx(30.0)


def test_eia930_long_fuel_rows_are_pivoted():
    frame = pd.DataFrame(
        {
            "period": ["2026-07-13T01:00:00Z"] * 3,
            "respondent": ["ERCO"] * 3,
            "type-name": ["Natural gas", "Coal", "Nuclear"],
            "value": [30_000.0, 10_000.0, 5_000.0],
        }
    )
    result = normalize_eia930(frame)
    assert result.loc[0, "gas_generation_actual_mw"] == pytest.approx(30_000.0)
    assert result.loc[0, "coal_generation_actual_mw"] == pytest.approx(10_000.0)
    assert result.loc[0, "nuclear_actual_mw"] == pytest.approx(5_000.0)


def test_eia930_total_interchange_is_converted_to_net_imports():
    frame = pd.DataFrame(
        {
            "period": ["2026-07-13T01:00:00Z"],
            "respondent": ["ERCO"],
            "type-name": ["Total interchange"],
            "value": [1_250.0],
        }
    )

    result = normalize_eia930(frame)

    assert result.loc[0, "net_imports_actual_mw"] == pytest.approx(-1_250.0)


def test_explicit_utc_hours_survive_dst_fall_back_as_unique_rows():
    frame = pd.DataFrame(
        {
            "valid_at": [
                "2026-11-01T01:00:00-05:00",
                "2026-11-01T01:00:00-06:00",
            ],
            "issued_at": ["2026-10-31T12:00:00Z"] * 2,
            "baseline_mw": [50_000.0, 49_000.0],
        }
    )
    result = normalize_component_product(
        frame,
        component="load",
        product_id="NP3-560-CD",
        retrieved_at="2026-10-31T12:01:00Z",
    )
    local = result["valid_at"].dt.tz_convert("America/Chicago")
    assert local.dt.hour.tolist() == [1, 1]
    assert result["valid_at"].is_unique
