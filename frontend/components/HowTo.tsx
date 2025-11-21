"use client";
import React from "react";

export default function HowTo() {
    return (
        <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">How to test </h3>
            <ol className="list-decimal list-inside text-gray-700">
                
                <li>Click <b>Ingest sample docs</b>.</li>
                <li>Ask: Can a customer return a damaged blender after 20 days?</li>
                <li>Ask: What's the shipping SLA to East Malaysia for bulky items?</li>
                <li>Ask: What couriers use for shipping?</li>
                <li>Ask: What is the warranty period for PowerBlend 100?</li>
            </ol>
        </div>
    );
}
