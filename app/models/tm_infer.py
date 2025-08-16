%%bash
BACKEND_URL="https://cheranengg-dhf-backend-fastapi-space.hf.space"
TOKEN="my-strong-secret"

read -r -d '' REQ <<'JSON'
{
  "requirements": [
    { "Requirement ID": "PR-001", "Requirements": "Functional Requirements" },
    { "Requirement ID": "PR-002", "Requirements": "Pump shall detect air-in-line within 1 second" }
  ],
  "ha": [
    { "requirement_id": "PR-002", "risk_id": "HA-001", "risk_to_health": "Air embolism",
      "risk_control": "Air-in-line detection alarm" }
  ],
  "dvp": [
    { "verification_id": "DV-001", "requirement_id": "PR-002",
      "verification_method": "Physical testing",
      "test_procedure": "- Introduce 50 µL air bolus\n- Verify alarm <= 1 s",
      "acceptance_criteria": "Alarm within 1 s" }
  ]
}
JSON

curl -sS "$BACKEND_URL/tm?limit=10" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary "$REQ" \
  | python -m json.tool
