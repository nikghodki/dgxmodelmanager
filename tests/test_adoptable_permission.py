"""Test GET /api/containers/adoptable requires admin."""


async def test_adoptable_requires_admin(api_stack, make_user):
    """GET /api/containers/adoptable is 403 for viewer, 200 for admin."""
    # Create viewer and admin users
    make_user("viewer", "viewer123", "viewer")
    make_user("admin", "admin123", "admin")

    # Login as viewer
    resp = api_stack.post("/dgxmm/api/auth/login", json={"username": "viewer", "password": "viewer123"})
    viewer_csrf = resp.json()["csrf_token"]

    # Viewer should get 403
    resp = api_stack.get("/dgxmm/api/containers/adoptable", headers={"X-CSRF-Token": viewer_csrf})
    assert resp.status_code == 403

    # Login as admin
    resp = api_stack.post("/dgxmm/api/auth/login", json={"username": "admin", "password": "admin123"})
    admin_csrf = resp.json()["csrf_token"]

    # Admin should get 200
    resp = api_stack.get("/dgxmm/api/containers/adoptable", headers={"X-CSRF-Token": admin_csrf})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
