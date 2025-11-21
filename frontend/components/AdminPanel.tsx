"use client";
import React from "react";
import { apiIngest, apiMetrics, apiHealth } from "@/lib/api";
import Metrics from "@/components/Metrics";

export default function AdminPanel() {
    const [health, setHealth] = React.useState<any>(null);
    const [metrics, setMetrics] = React.useState<any>(null);
    const [busy, setBusy] = React.useState(false);

    const refresh = async () => {
        const m = await apiMetrics();
        setMetrics(m);
    };

    const getHealth = async () => {
        const h = await apiHealth();
        setHealth(h);
    };

    const ingest = async () => {
        setBusy(true);
        try {
            await apiIngest();
            await refresh();
        } finally {
            setBusy(false);
        }
    };

    React.useEffect(() => {
        getHealth();
        refresh();
    }, []);

    return (
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
                Admin Panel
            </h2>

            <div className="flex flex-col sm:flex-row gap-3 mb-6">
                <button
                    onClick={ingest}
                    disabled={busy}
                    className="button-primary"
                >
                    {busy ? "Indexing..." : "Ingest sample docs"}
                </button>
                <button onClick={refresh} className="button-secondary">
                    Refresh metrics
                </button>
            </div>

            <Metrics metrics={metrics} health={health} />
        </div>
    );
}
