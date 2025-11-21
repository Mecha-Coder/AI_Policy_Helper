"use client";
import React from "react";

export default function Info({
    metrics,
    health,
}: {
    metrics: any;
    health: any;
}) {
    console.log("Metrics component render");
    console.log(metrics);
    console.log("Health status:");
    console.log(health);
    return (
        <div className="border-t border-gray-200 pt-4">
            <h3 className="text-lg font-medium text-gray-700 mb-3">Metrics</h3>

            <div className="space-y-2">
                <div className="flex justify-between">
                    <span className="text-gray-600">Health:</span>

                    {health ? (
                        <span className="font-medium text-green-600">
                            Healthy
                        </span>
                    ) : (
                        <span className="font-medium text-red-600">
                            Unhealthy
                        </span>
                    )}
                </div>

                {metrics &&
                    Object.entries(metrics).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                            <span className="text-gray-600">{key}:</span>
                            <span className="font-medium">
                                {value as string}
                            </span>
                        </div>
                    ))}
            </div>
        </div>
    );
}
